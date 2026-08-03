// Copyright 2026, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Implementation of the rerun recorder logic for the sensor fusion.
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup tracking
 */

#include "sensor_fusion_rerun.hpp"

#include "math/m_api.h"
#include "math/m_vec3.h"

#include "tracking/t_camera_models.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <vector>


using namespace xrt::auxiliary::tracking;
using namespace xrt::tracking::constellation;
using namespace xrt::tracking::constellation::optimizer;
using namespace xrt::tracking::constellation::optimizer::sensor_fusion;

// Anonymous namespace for internal functions.
namespace {

/*
 *
 * Helper functions
 *
 */

const char *
trackingStateName(DeviceTrackingState state)
{
	switch (state) {
	case DeviceTrackingState::Uninitialized: return "Uninitialized";
	case DeviceTrackingState::Tracking: return "Tracking";
	case DeviceTrackingState::Lost: return "Lost";
	default: return "Unknown";
	}
}

/*!
 * How far the solved pose lands from the blobs this camera actually measured, in pixels.
 *
 * This is the camera residual the solve minimized with the whitening taken back off, so it reads directly as "the LEDs
 * landed this far from where they were seen" rather than in the standard deviations the solver works in.
 */
float
reprojectionErrorPixels(const SnapshotObservation &observation, const xrt_pose &Tcv_world_device)
{
	const xrt_pose Tcv_cam_device = predictedCameraDevicePose(observation.Tcv_world_cam, Tcv_world_device);

	double sum_squared_error = 0.0;
	size_t num_projected = 0;

	for (size_t i = 0; i < observation.num_points; i++) {
		xrt_vec3 T_cam_led;
		math_quat_rotate_vec3(&Tcv_cam_device.orientation, &observation.points3d[i], &T_cam_led);
		T_cam_led = T_cam_led + Tcv_cam_device.position;

		if (T_cam_led.z <= 0.0f) {
			// Behind the camera, so there is no pixel to compare against.
			continue;
		}

		float projected_x = 0.0f;
		float projected_y = 0.0f;
		if (!camera_models::project<float>(observation.params, T_cam_led.x, T_cam_led.y, T_cam_led.z,
		                                   projected_x, projected_y)) {
			continue;
		}

		const float error_x = projected_x - observation.points2d[i].x;
		const float error_y = projected_y - observation.points2d[i].y;

		sum_squared_error += (error_x * error_x) + (error_y * error_y);
		num_projected++;
	}

	if (num_projected == 0) {
		return 0.0f;
	}

	return static_cast<float>(std::sqrt(sum_squared_error / static_cast<double>(num_projected)));
}

//! The angle between two orientations, in degrees.
float
vec3Length(const xrt_vec3_f64 &v)
{
	return static_cast<float>(std::sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z)));
}

/*
 *
 * Common entity names
 *
 */

std::string
getCameraObservationEntityName(size_t mosaic_idx, size_t camera_idx, t_constellation_device_id_t device_id)
{
	return std::format("{}/observed/{}", getWorldCameraEntityName(mosaic_idx, camera_idx), device_id);
}

//! Lives in world space rather than under the camera, since the LEDs are placed by the device's solved pose.
std::string
getMatchedLedsEntityName(size_t mosaic_idx, size_t camera_idx, t_constellation_device_id_t device_id)
{
	return std::format("{}/matched_leds/{}/{}/{}", getWorldEntityName(), mosaic_idx, camera_idx, device_id);
}

std::string
getDeviceEntityName(t_constellation_device_id_t device_id)
{
	return std::format("{}/devices/{}", getWorldEntityName(), device_id);
}

//! Sibling of the device entity, so that it is not moved around by the device's per-solve transform.
std::string
getDeviceWindowEntityName(t_constellation_device_id_t device_id)
{
	return std::format("{}/window/{}", getWorldEntityName(), device_id);
}

std::string
getSummaryEntityName()
{
	return "summary";
}

std::string
getSolverMetricsEntityName()
{
	return "metrics/solver";
}

std::string
getDeviceMetricsEntityName(t_constellation_device_id_t device_id)
{
	return std::format("metrics/devices/{}", device_id);
}

std::string
getCameraMetricsEntityName(size_t mosaic_idx, size_t camera_idx, t_constellation_device_id_t device_id)
{
	return std::format("metrics/cameras/{}/{}/{}", mosaic_idx, camera_idx, device_id);
}

std::string
getSolveTimelineName()
{
	return "solves";
}

std::string
getTimeTimelineName()
{
	return "fusion_time";
}

}; // namespace

namespace xrt::tracking::constellation::optimizer::sensor_fusion {

/*
 *
 * Private methods
 *
 */

void
RerunContext::logStaticScene()
{
	// The whole solve happens in OpenCV space.
	this->stream->log_static("/", rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN);
	this->stream->log_static(getWorldEntityName(), rerun::TransformAxes3D(kAxisLength));
}

void
RerunContext::logSolveSummary(const SolveSnapshot &snapshot)
{
	std::string text = std::format(        //
	    "# Sensor fusion solve {}\n"       //
	    "\n"                               //
	    "* Keyframes in window: {}\n"      //
	    "* Parameter blocks: {}\n"         //
	    "* Residuals: {}\n"                //
	    "* Iterations: {}\n"               //
	    "* Termination: {}\n"              //
	    "* Solve time: {:.1f} ms\n"        //
	    "* Gravity: {:.4f} m/s^2\n"        //
	    "* Late samples dropped: {}\n"     //
	    "\n"                               //
	    "## Fit\n"                         //
	    "\n"                               //
	    "* Initial RMS: {:.3f} sigma\n"    //
	    "* Final RMS: {:.3f} sigma\n"      //
	    "\n"                               //
	    "## Devices\n"                     //
	    "\n",                              //
	    snapshot.solve_index,              //
	    snapshot.keyframes.size(),         //
	    snapshot.num_parameter_blocks,     //
	    snapshot.num_residuals,            //
	    snapshot.num_iterations,           //
	    snapshot.termination,              //
	    snapshot.solve_time_s * 1000.0,    //
	    snapshot.world_gravity_mag,        //
	    snapshot.num_late_samples_dropped, //
	    snapshot.initial_rms_sigma,        //
	    snapshot.final_rms_sigma);         //

	for (const SnapshotDevice &device : snapshot.devices) {
		if (device.id == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
			continue;
		}

		text += std::format(                                         //
		    "* Device {}: {}, {} observed keyframes, {} in solve\n", //
		    device.id,                                               //
		    trackingStateName(device.tracking_state),                //
		    device.num_observed_keyframes,                           //
		    std::popcount(device.keyframe_mask));                    //
	}

	this->stream->log(getSummaryEntityName(),
	                  rerun::TextDocument(text).with_media_type(rerun::MediaType::markdown()));
}

void
RerunContext::logSolverMetrics(const SolveSnapshot &snapshot)
{
	const std::string metrics = getSolverMetricsEntityName();

	// The two RMS numbers are the whole story of a solve: where the seeds put it and where it got to. Both are in
	// standard deviations, so they stay comparable as the window's residual count moves around.
	this->stream->log(metrics + "/initial_rms_sigma", rerun::Scalars(snapshot.initial_rms_sigma));
	this->stream->log(metrics + "/final_rms_sigma", rerun::Scalars(snapshot.final_rms_sigma));
	this->stream->log(metrics + "/iterations", rerun::Scalars(static_cast<double>(snapshot.num_iterations)));
	this->stream->log(metrics + "/solve_time_ms", rerun::Scalars(snapshot.solve_time_s * 1000.0));
	this->stream->log(metrics + "/residuals", rerun::Scalars(static_cast<double>(snapshot.num_residuals)));
	this->stream->log(metrics + "/parameter_blocks",
	                  rerun::Scalars(static_cast<double>(snapshot.num_parameter_blocks)));
	this->stream->log(metrics + "/keyframes", rerun::Scalars(static_cast<double>(snapshot.num_keyframes)));
	this->stream->log(metrics + "/gravity_m_s2", rerun::Scalars(snapshot.world_gravity_mag));
	this->stream->log(metrics + "/late_samples_dropped",
	                  rerun::Scalars(static_cast<double>(snapshot.num_late_samples_dropped)));
}

void
RerunContext::logCameras(const SolveSnapshot &snapshot)
{
	/*
	 * Camera poses arrive on the observations rather than from a fixed table: the cameras are on a headset that
	 * moves, so where a camera was is a property of the exposure, not of the rig. Only the newest keyframe's view
	 * is drawn, the older ones would stack onto the same entities and only the last would survive.
	 */
	for (uint32_t i = 0; i < snapshot.keyframes[0].num_observations; i++) {
		const SnapshotObservation &observation = snapshot.keyframes[0].observations[i];
		const std::string camera_entity =
		    getWorldCameraEntityName(observation.mosaic_idx, observation.camera_idx);

		this->stream->log(camera_entity, toRerunTransform(observation.Tcv_world_cam));
		this->stream->log(camera_entity + "/axes", rerun::TransformAxes3D(kAxisLength));
	}
}

void
RerunContext::logDevice(const SolveSnapshot &snapshot, uint32_t device_fusion_idx)
{
	const SnapshotDevice &device = snapshot.devices[device_fusion_idx];
	const SolvedDeviceKeyframe &newest = snapshot.keyframes[0].devices[device_fusion_idx];

	const std::string device_entity = getDeviceEntityName(device.id);

	if (!newest.in_solve) {
		// The solve reached this device somewhere in the window but not at its newest keyframe, so there is no
		// current pose to draw. Clearing beats leaving the last one up and having it read as live.
		this->stream->log(device_entity, rerun::Clear::RECURSIVE);
		return;
	}

	this->stream->log(device_entity, toRerunTransform(newest.Tcv_world_device));
	this->stream->log(device_entity + "/axes", rerun::TransformAxes3D(kAxisLength));

	// The velocity is in world space but the device entity carries the device's own transform, so the arrow is
	// drawn from the window entity, which does not move.
	this->stream->log(                                                             //
	    getDeviceWindowEntityName(device.id) + "/velocity",                        //
	    rerun::Arrows3D::from_vectors(                                             //
	        {rerun::Vector3D(static_cast<float>(newest.velocity.x),                //
	                         static_cast<float>(newest.velocity.y),                //
	                         static_cast<float>(newest.velocity.z))})              //
	        .with_origins({rerun::Position3D(newest.Tcv_world_device.position.x,   //
	                                         newest.Tcv_world_device.position.y,   //
	                                         newest.Tcv_world_device.position.z)}) //
	        .with_colors({deviceColor(device.id, 1.0f)})                           //
	);                                                                             //
}

void
RerunContext::logDeviceWindow(const SolveSnapshot &snapshot, uint32_t device_fusion_idx)
{
	if (!kLogOldKeyframes) {
		return;
	}

	const SnapshotDevice &device = snapshot.devices[device_fusion_idx];

	std::vector<rerun::Position3D> positions;
	std::vector<rerun::Position3D> strip_points;

	// Oldest first, so the strip runs the way the device actually travelled.
	for (size_t i = snapshot.keyframes.size(); i-- > 0;) {
		const SolvedDeviceKeyframe &solved = snapshot.keyframes[i].devices[device_fusion_idx];

		if (!solved.in_solve) {
			continue;
		}

		const xrt_vec3 &position = solved.Tcv_world_device.position;
		positions.emplace_back(position.x, position.y, position.z);
		strip_points.emplace_back(position.x, position.y, position.z);
	}

	const std::string window_entity = getDeviceWindowEntityName(device.id);

	if (positions.empty()) {
		this->stream->log(window_entity + "/line", rerun::Clear::RECURSIVE);
		this->stream->log(window_entity + "/points", rerun::Clear::RECURSIVE);
		return;
	}

	// This is the window, not a trajectory: every state in it is still being solved for, and re-logging the whole
	// thing each solve is what makes the older states visibly settle as they pick up more constraints.
	this->stream->log(                                          //
	    window_entity + "/line",                                //
	    rerun::LineStrips3D({rerun::LineStrip3D(strip_points)}) //
	        .with_radii({kTrajectoryLineRadiusM})               //
	        .with_colors({deviceColor(device.id, 0.5f)})        //
	);                                                          //
	this->stream->log(                                          //
	    window_entity + "/points",                              //
	    rerun::Points3D(positions)                              //
	        .with_radii({kTrajectoryPointRadiusM})              //
	        .with_colors({deviceColor(device.id, 1.0f)})        //
	);                                                          //
}

void
RerunContext::logDeviceMetrics(const SolveSnapshot &snapshot, uint32_t device_fusion_idx)
{
	const SnapshotDevice &device = snapshot.devices[device_fusion_idx];
	const SolvedDeviceKeyframe &newest = snapshot.keyframes[0].devices[device_fusion_idx];

	const std::string metrics = getDeviceMetricsEntityName(device.id);

	this->stream->log(metrics + "/observed_keyframes",
	                  rerun::Scalars(static_cast<double>(device.num_observed_keyframes)));
	this->stream->log(metrics + "/states_in_solve",
	                  rerun::Scalars(static_cast<double>(std::popcount(device.keyframe_mask))));
	this->stream->log(metrics + "/tracking_state",
	                  rerun::Scalars(static_cast<double>(static_cast<int>(device.tracking_state))));

	if (!newest.in_solve) {
		return;
	}

	this->stream->log(metrics + "/speed_m_s", rerun::Scalars(vec3Length(newest.velocity)));

	// The biases are what the IMU factors are really solving for, and they are the first thing to go somewhere
	// unphysical when the window is too weak to constrain them.
	this->stream->log(metrics + "/accel_bias_m_s2", rerun::Scalars(vec3Length(newest.accel_bias)));
	this->stream->log(metrics + "/gyro_bias_rad_s", rerun::Scalars(vec3Length(newest.gyro_bias)));

	// Per-axis, so these can be read straight against what the offline calibration reports.
	this->stream->log(metrics + "/accel_bias_body_x", rerun::Scalars(newest.accel_bias.x));
	this->stream->log(metrics + "/accel_bias_body_y", rerun::Scalars(newest.accel_bias.y));
	this->stream->log(metrics + "/accel_bias_body_z", rerun::Scalars(newest.accel_bias.z));

	/*
	 * The same bias rotated into the world.
	 *
	 * Accelerometer bias is a body-frame quantity, so a correct one is constant in the body frame and swings in
	 * the world as the device turns. Anything the bias is absorbing that is really world-frame - a gravity
	 * direction the solve cannot express, say - behaves the other way round: steady in the world and swinging in
	 * the body. Logging both is what makes those two tell each other apart, since either one on its own just looks
	 * like "the bias moves when I move the device".
	 */
	const xrt_vec3 body_bias{static_cast<float>(newest.accel_bias.x), //
	                         static_cast<float>(newest.accel_bias.y), //
	                         static_cast<float>(newest.accel_bias.z)};

	xrt_vec3 world_bias;
	math_quat_rotate_vec3(&newest.Tcv_world_device.orientation, &body_bias, &world_bias);

	this->stream->log(metrics + "/accel_bias_world_x", rerun::Scalars(world_bias.x));
	this->stream->log(metrics + "/accel_bias_world_y", rerun::Scalars(world_bias.y));
	this->stream->log(metrics + "/accel_bias_world_z", rerun::Scalars(world_bias.z));
	this->stream->log(metrics + "/accel_scale_x", rerun::Scalars(newest.accel_scale.x));
	this->stream->log(metrics + "/accel_scale_y", rerun::Scalars(newest.accel_scale.y));
	this->stream->log(metrics + "/accel_scale_z", rerun::Scalars(newest.accel_scale.z));

	// The extrinsic is held identity-seeded and, for now, is expected to stay near it. A drift here is either a
	// real calibration or a sign the solve is absorbing something else into it.
	this->stream->log(metrics + "/imu_model_offset_deg",
	                  rerun::Scalars(orientationErrorDegrees(XRT_QUAT_IDENTITY, device.Qcv_imu_model)));
}

void
RerunContext::logObservations(const SolveSnapshot &snapshot)
{
	// Only the newest keyframe: the older ones in the window were drawn on the solves they were newest for, and
	// stacking every keyframe's blobs would bury the current one.
	const SnapshotKeyframe &keyframe = snapshot.keyframes[0];

	for (uint32_t i = 0; i < keyframe.num_observations; i++) {
		const SnapshotObservation &observation = keyframe.observations[i];
		const SnapshotDevice &device = snapshot.devices[observation.device_idx];
		const SolvedDeviceKeyframe &solved = keyframe.devices[observation.device_idx];

		const std::string observation_entity =
		    getCameraObservationEntityName(observation.mosaic_idx, observation.camera_idx, device.id);
		const std::string matched_leds_entity =
		    getMatchedLedsEntityName(observation.mosaic_idx, observation.camera_idx, device.id);

		if (!solved.in_solve) {
			// Observed, but the solve holds no state for it here, so there is nothing to compare against.
			this->stream->log(observation_entity, rerun::Clear::RECURSIVE);
			this->stream->log(matched_leds_entity, rerun::Clear::RECURSIVE);
			continue;
		}

		// Where the solve says this camera is looking at the device, as a pose under the camera entity.
		const xrt_pose Tcv_cam_device =
		    predictedCameraDevicePose(observation.Tcv_world_cam, solved.Tcv_world_device);

		this->stream->log(observation_entity, toRerunTransform(Tcv_cam_device));
		this->stream->log(observation_entity + "/axes", rerun::TransformAxes3D(kAxisLength));

		// The LEDs this camera matched, placed in the world by the solved pose. Where these sit relative to the
		// device's model is the visual form of the reprojection error logged below.
		std::vector<rerun::Position3D> led_positions;
		led_positions.reserve(observation.num_points);

		for (uint32_t i = 0; i < observation.num_points; i++) {
			const xrt_vec3 &T_model_led = observation.points3d[i];
			xrt_vec3 T_world_led;
			math_quat_rotate_vec3(&solved.Tcv_world_device.orientation, &T_model_led, &T_world_led);
			T_world_led = T_world_led + solved.Tcv_world_device.position;

			led_positions.emplace_back(T_world_led.x, T_world_led.y, T_world_led.z);
		}

		this->stream->log(                                                                        //
		    matched_leds_entity,                                                                  //
		    rerun::Points3D(led_positions)                                                        //
		        .with_radii({kLedPointRadiusM})                                                   //
		        .with_colors({cameraColor(observation.mosaic_idx, observation.camera_idx, 1.0f)}) //
		);                                                                                        //

		const std::string camera_metrics =
		    getCameraMetricsEntityName(observation.mosaic_idx, observation.camera_idx, device.id);

		this->stream->log(camera_metrics + "/reprojection_error_px",
		                  rerun::Scalars(reprojectionErrorPixels(observation, solved.Tcv_world_device)));
		this->stream->log(camera_metrics + "/correspondence_count",
		                  rerun::Scalars(static_cast<double>(observation.num_points)));

		// How far the solve drifted from the per-frame pose the blob matcher recovered on its own. The solve is
		// fit to the blobs rather than to this, so a large disagreement is a real signal rather than a
		// residual.
		const float position_error_m =
		    m_vec3_len(Tcv_cam_device.position - observation.Tcv_cam_device_seed.position);
		const float orientation_error_deg =
		    orientationErrorDegrees(observation.Tcv_cam_device_seed.orientation, Tcv_cam_device.orientation);

		this->stream->log(camera_metrics + "/seed_position_error_m", rerun::Scalars(position_error_m));
		this->stream->log(camera_metrics + "/seed_orientation_error_deg",
		                  rerun::Scalars(orientation_error_deg));
	}
}

/*
 *
 * Public methods
 *
 */

void
RerunContext::logSolve(const SolveSnapshot &snapshot)
{
	// A solve with an empty window has nothing to say, and every helper below indexes keyframe 0.
	if (snapshot.keyframes.empty()) {
		return;
	}

	this->logStaticScene();

	// The sequence is what advances on every solve; the timestamp is the exposure the newest keyframe came from.
	// Keeping both means the same recording can be read either as "one step per solve" or against wall time,
	// which matters because solves are not evenly spaced.
	this->stream->set_time_timestamp_nanos_since_epoch(getTimeTimelineName(), snapshot.keyframes[0].timestamp_ns);
	this->stream->set_time_sequence(getSolveTimelineName(), static_cast<int64_t>(snapshot.solve_index));

	this->logSolveSummary(snapshot);
	this->logSolverMetrics(snapshot);
	this->logCameras(snapshot);
	this->logObservations(snapshot);

	for (uint32_t device_fusion_idx = 0; device_fusion_idx < snapshot.devices.size(); device_fusion_idx++) {
		const SnapshotDevice &device = snapshot.devices[device_fusion_idx];

		if (!device.in_solve) {
			continue;
		}

		this->logDevice(snapshot, device_fusion_idx);
		this->logDeviceWindow(snapshot, device_fusion_idx);
		this->logDeviceMetrics(snapshot, device_fusion_idx);
	}

	this->stream->reset_time();
}

}; // namespace xrt::tracking::constellation::optimizer::sensor_fusion
