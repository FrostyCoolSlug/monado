// Copyright 2026, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Constellation tracker sensor fusion.
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup tracking
 */

#include "constellation/t_constellation_tracker_internal.hpp"

#include "math/m_eigen_interop.hpp"

#include "internal_math.hpp"
#include "sensor_fusion.hpp"

#ifdef XRT_FEATURE_RERUN
#include "sensor_fusion_rerun.hpp"
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <thread>


namespace {

DEBUG_GET_ONCE_LOG_OPTION(sensor_fusion_log, "T_CONSTELLATION_SENSOR_FUSION_LOG", U_LOGGING_WARN)
DEBUG_GET_ONCE_BOOL_OPTION(sensor_fusion_enable_rerun, "T_CONSTELLATION_SENSOR_FUSION_RERUN_ENABLE", false)
DEBUG_GET_ONCE_BOOL_OPTION(sensor_fusion_rerun_spawn, "T_CONSTELLATION_SENSOR_FUSION_RERUN_SPAWN", true)

#define SF_TRACE(sensor_fusion, ...) U_LOG_IFL_T(sensor_fusion->log_level, __VA_ARGS__)
#define SF_DEBUG(sensor_fusion, ...) U_LOG_IFL_D(sensor_fusion->log_level, __VA_ARGS__)
#define SF_INFO(sensor_fusion, ...) U_LOG_IFL_I(sensor_fusion->log_level, __VA_ARGS__)
#define SF_WARN(sensor_fusion, ...) U_LOG_IFL_W(sensor_fusion->log_level, __VA_ARGS__)
#define SF_ERROR(sensor_fusion, ...) U_LOG_IFL_E(sensor_fusion->log_level, __VA_ARGS__)

using namespace xrt::tracking;
using namespace xrt::tracking::constellation::sensor_fusion;

using Eigen::MatrixXd;
using Eigen::Vector;
using Eigen::Vector3;
using Eigen::VectorXd;

//! Ceres hands its jacobian blocks out row-major, which is not Eigen's default for a dynamically sized matrix.
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXd;

bool
constellationDeviceToFusionDevice(const DeviceList &device_list,
                                  t_constellation_device_id_t constellation_id,
                                  uint32_t &fusion_device_idx)
{
	for (uint32_t i = 0; i < device_list.size(); i++) {
		const auto &device = device_list[i];

		/*
		 * Relaxed is enough here: everything reaching this helper already holds the fusion lock, which is what
		 * orders it against registration.
		 */
		const t_constellation_device_id_t slot_id = device.id.load(std::memory_order_relaxed);

		// Skip empty slots, so an unoccupied slot can never be matched against a real device ID
		if (slot_id == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
			continue;
		}

		if (slot_id == constellation_id) {
			fusion_device_idx = i;
			return true;
		}
	}

	return false;
}

void
writeSampleToKeyframe(SensorFusion *fusion,
                      const constellation::CameraSample &sample,
                      Keyframe &keyframe,
                      uint32_t keyframe_age)
{
	xrt_pose Tcv_world_cam = sample.Txr_world_cam.value();
	math_pose_convert_from_opencv(&Tcv_world_cam, &Tcv_world_cam);

	for (uint32_t i = 0; i < sample.device_count; i++) {
		const auto &device = sample.device_states[i];

		if (!device.found_pose.has_value()) {
			continue;
		}

		// Get the fusion ID
		uint32_t fusion_device_idx;
		if (!constellationDeviceToFusionDevice(fusion->devices, device.device_id, fusion_device_idx)) {
			// Device not added/has been removed
			continue;
		}

		const auto led_model = fusion->devices[fusion_device_idx].led_model;

		/*
		 * A keyframe has room for every camera of a mosaic to observe every device exactly once, which is all
		 * that can legitimately land in one. Should never happen, but overrunning would scribble straight over
		 * the rest of the fusion state.
		 */
		assert(keyframe.num_observations < keyframe.observations.size());
		if (keyframe.num_observations >= keyframe.observations.size()) {
			SF_WARN(fusion, "Keyframe at %" PRIi64 " is full at %zu observations, dropping camera %u's.",
			        keyframe.timestamp_ns, keyframe.observations.size(), sample.camera_index);
			return;
		}

		auto &observation = keyframe.observations[keyframe.num_observations++] = {
		    .camera_idx = sample.camera_index,
		    .device_idx = fusion_device_idx,
		    .Tcv_world_cam = Tcv_world_cam,
		    .Tcv_cam_device_seed = device.found_pose->Tcv_cam_device,
		    .points2d = {},
		    .points3d = {},
		    .num_points = 0,
		};

		for (uint32_t bi = 0; bi < sample.blob_count; bi++) {
			const auto &blob = sample.blobs[bi];

			if (blob.matched_device_id == device.device_id) {
				int32_t led_idx = findLedIndexById(led_model, blob.matched_device_led_id);

				// Ignore blob
				if (led_idx == -1) {
					continue;
				}

				if (kOptimizeUndistortedPoints) {
					observation.points2d[observation.num_points] = blob.center_undistorted;
				} else {
					observation.points2d[observation.num_points] = blob.center_distorted;
				}

				observation.points3d[observation.num_points] = led_model->leds[led_idx].position;
				observation.num_points += 1;
			}
		}

		if (observation.num_points == 0) {
			keyframe.num_observations -= 1;
			continue;
		}

		/*
		 * First sighting of this device in this keyframe, so the keyframe now counts towards the device's
		 * window coverage.
		 */
		const uint32_t device_bit = 1u << fusion_device_idx;
		if ((keyframe.observed_device_mask & device_bit) == 0) {
			keyframe.observed_device_mask |= device_bit;
			fusion->devices[fusion_device_idx].num_observed_keyframes += 1;

			/*
			 * The device's IMU chain only has states where it was actually seen, so this keyframe
			 * gaining one changes which intervals exist around it.
			 */
			fusion->dirtyPreintegrationsForNewObservationLocked(fusion_device_idx, keyframe_age);
		}
	}
}

//! How many times a collect will retry after being lapped by the IMU thread before it gives up.
constexpr uint32_t kImuCollectAttempts = 4;

/*!
 * Velocity seeds above this are taken as a failed PnP rather than a fast device, and dropped in favour of zero. A
 * tracked controller tops out a long way under this even when thrown (please don't try to prove me wrong).
 */
constexpr double kMaxSeedVelocityMetersPerSecond = 10.0;

constexpr bool kOptimizeWorldGravity = false;

/*!
 * Whether to exit early if we hit an RMS floor. Disabled for now since we don't have our noise model tweaked very well.
 */
constexpr bool kExitOnRmsFloor = false;

/*!
 * The residual RMS, in standard deviations, at which the solve is called good enough and stopped.
 *
 * Every residual in the problem is already divided through by its own noise model, so one RMS across all of them is
 * dimensionless and means the same thing whatever the window happens to hold.
 *
 * The camera factors carry a Huber loss, so what the callback compares against is the robustified cost rather than
 * the raw sum of squares.
 *
 * @todo Measure the real floor and set this from it. Too low and the solve never exits early, falling back on
 *       `max_num_iterations`; too high and it stops while the fit is still visibly wrong.
 */
constexpr double kTargetResidualRmsSigma = 0.5;

enum class ImuCollectResult
{
	//! Samples were collected successfully.
	Ok,
	//! The IMU thread overwrote slots while we were reading them, so the collected samples are untrustworthy.
	Lapped,
	//! More samples fall in the range than the caller's staging buffer can hold.
	Overflow,
	//! No sample newer than `end_time_ns` has arrived yet, so the range is not closed off.
	NoSampleAfterEnd,
};

/*!
 * One attempt at collecting `device`'s buffered IMU samples falling in the range `(start_time_ns, end_time_ns]` into
 * `out_samples`, oldest first, along with the first sample newer than `end_time_ns`.
 *
 * @note Runs concurrently with `pushImuSample()`, so it can come back lapped; use `collectImuSamplesForRange()` rather
 *       than calling this directly.
 */
ImuCollectResult
tryCollectImuSamplesForRange(Device &device,
                             timepoint_ns start_time_ns,
                             timepoint_ns end_time_ns,
                             std::span<xrt_imu_sample> out_samples,
                             uint32_t &out_num_samples,
                             xrt_imu_sample &out_first_after)
{
	/*
	 * Acquire, pairing with the release store in pushImuSample(): every slot below this count has been fully
	 * written and is visible to us.
	 */
	const uint64_t write_count = device.imu_write_count.load(std::memory_order_acquire);

	/*
	 * Absolute index of the oldest sample the ring still holds. Before the ring has filled for the first time this
	 * is 0, so we never read a slot that has not been written yet and the buffer needs no zero-initialization.
	 */
	const uint64_t oldest = write_count - std::min<uint64_t>(write_count, kImuBufferSize);

	// The lowest absolute index we actually dereference, used to bound the lap check below.
	uint64_t lowest_read = write_count;

	const auto slot_at = [&](uint64_t abs) -> const RingImuSample & {
		lowest_read = std::min(lowest_read, abs);
		return device.imu_samples[abs % kImuBufferSize];
	};

	/*
	 * Samples ascend in time with their absolute index, so the window is a contiguous run of indices and its start
	 * can be found by bisection. This keeps us from touching thousands of slots we do not need, which matters less
	 * for speed than for the lap check: the fewer slots we read, the narrower the window in which the IMU thread
	 * can invalidate them.
	 */
	uint64_t lo = oldest;
	uint64_t hi = write_count;
	while (lo < hi) {
		const uint64_t mid = lo + ((hi - lo) / 2);

		/*
		 * Only the timestamp is needed to steer the bisection, and a probe that reads one word instead of the
		 * whole sample is one fewer chance for the writer to tear something underneath us.
		 */
		if (slot_at(mid).loadTimestamp() <= start_time_ns) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}

	out_num_samples = 0;

	bool have_first_after = false;
	for (uint64_t abs = lo; abs < write_count; abs++) {
		const xrt_imu_sample sample = slot_at(abs).load();

		if (sample.timestamp_ns > end_time_ns) {
			/*
			 * Ascending timestamps mean the first sample past the end is the one preintegrate() wants to
			 * interpolate the tail of the range with.
			 */
			out_first_after = sample;
			have_first_after = true;
			break;
		}

		if (out_num_samples >= out_samples.size()) {
			out_num_samples = 0;
			return ImuCollectResult::Overflow;
		}

		out_samples[out_num_samples++] = sample;
	}

	/*
	 * The IMU thread writes slots without waiting for us, so anything read above may have been overwritten
	 * underneath us, possibly even a torn read, where some field are from a newer sample, since a sample is wider
	 * than a single store. Slots from `write_count_after - kImuBufferSize` up are the ones still intact, so if the
	 * writer got past the oldest index we touched then what we collected is garbage and has to be thrown away
	 * rather than fed to the optimizer.
	 */
	const uint64_t write_count_after = device.imu_write_count.load(std::memory_order_acquire);
	if (write_count_after > lowest_read + kImuBufferSize) {
		return ImuCollectResult::Lapped;
	}

	if (!have_first_after) {
		return ImuCollectResult::NoSampleAfterEnd;
	}

	return ImuCollectResult::Ok;
}

/*!
 * Collects `device`'s buffered IMU samples falling in the range `(start_time_ns, end_time_ns]` into `out_samples`,
 * oldest first so that preintegration reads linearly forward, and returns the first sample newer than `end_time_ns` in
 * `out_first_after`.
 */
ImuCollectResult
collectImuSamplesForRange(Device &device,
                          timepoint_ns start_time_ns,
                          timepoint_ns end_time_ns,
                          std::span<xrt_imu_sample> out_samples,
                          uint32_t &out_num_samples,
                          xrt_imu_sample &out_first_after)
{
	/*
	 * Being lapped is caused by fresh samples landing mid-collect, which is transient: a retry reads a window that
	 * has just moved on slightly and succeeds. Only a stalled fusion thread would see this repeat.
	 */
	for (uint32_t attempt = 0; attempt < kImuCollectAttempts; attempt++) {
		const ImuCollectResult result = tryCollectImuSamplesForRange( //
		    device,                                                   //
		    start_time_ns,                                            //
		    end_time_ns,                                              //
		    out_samples,                                              //
		    out_num_samples,                                          //
		    out_first_after);

		if (result != ImuCollectResult::Lapped) {
			return result;
		}
	}

	return ImuCollectResult::Lapped;
}

struct StaticCameraObservationCostFunctor
{
	camera_model params;
	DeviceObservation observation;
	Eigen::Transform<float, 3, Eigen::Isometry> Tcv_cam_world;

	template <typename T>
	bool
	operator()(const T *const keyframe_parameters,       //
	           const T *const imu_extrinsics_parameters, //
	           T *residuals) const                       //
	{
		const Pose<T> predicted_T_world_device{
		    Map<const Vector<T, Pose<T>::kNumParameters>>(keyframe_parameters)};

		const ImuExtrinsics<T> predicted_imu_extrinsics{
		    Map<const Vector<T, ImuExtrinsics<T>::kNumParameters>>(imu_extrinsics_parameters)};

		const auto predicted_T_camera_imu = Tcv_cam_world.cast<T>() * predicted_T_world_device.toTransform();

		const Quaternion<T> Q_cam_model =
		    Quaternion<T>(predicted_T_camera_imu.rotation()) * predicted_imu_extrinsics.Q_imu_model;

		for (size_t i = 0; i < observation.num_points; i++) {
			const auto blob_position_2d = map_vec2(this->observation.points2d[i]).cast<T>();
			const auto T_model_led = map_vec3(this->observation.points3d[i]).cast<T>();

			Map<Vector2<T>> residual(residuals + (i * kNumLedResiduals));
			computeLedResidual<T>(params,                               //
			                      predicted_T_camera_imu.translation(), //
			                      Q_cam_model,                          //
			                      blob_position_2d,                     //
			                      T_model_led,                          //
			                      residual);                            //

			residual /= T(kBlobPositionSigmaPixels);
		}

		return true;
	}
};

typedef ceres::AutoDiffCostFunction<StaticCameraObservationCostFunctor,       //
                                    ceres::DYNAMIC,                           //
                                    PoseWithVelocity<double>::kNumParameters, //
                                    ImuExtrinsics<double>::kNumParameters>    //
    StaticCameraObservationCostFunction;

/*!
 * Evaluates @p inner's residuals where the states actually are, but its jacobians at the points those states are
 * anchored at.
 *
 * Only the states a prior has covered are anchored. Everything else linearizes where it is, because nothing frozen
 * disagrees with it.
 *
 * Evaluating twice is the price: the residuals have to come from where the states are, since that is the error the
 * solve is there to drive down, while the jacobians have to come from the anchor, since that is the whole point.
 * What it buys is every factor over an anchored state agreeing with the prior about which linearization they are
 * all talking about, at the cost of jacobians a little staler than the ones a re-linearizing factor would use.
 */
class FirstEstimateJacobianCostFunction final : public ceres::CostFunction
{
public:
	//! The most parameter blocks any factor in this problem touches.
	static constexpr size_t kMaxParameterBlocks = 8;

	//! The widest ambient parameter block in this problem, @ref PoseWithVelocity.
	static constexpr int kMaxAmbientSize = 10;

	//! What one parameter block of the wrapped factor is linearized at.
	struct Anchor
	{
		//! The point to linearize this block at, or null to linearize it wherever it currently is.
		const double *linearization_point{nullptr};

		/*!
		 * The manifold the block is added to the problem with, or null for a block that lives in a plain
		 * Euclidean space.
		 *
		 * Has to be that same manifold: the lift in @ref Evaluate only cancels against its own plus
		 * jacobian.
		 */
		const ceres::Manifold *manifold{nullptr};
	};

	/*!
	 * @param inner   The factor to evaluate. Ownership is taken.
	 * @param anchors One per parameter block of @p inner, in the same order.
	 */
	FirstEstimateJacobianCostFunction(ceres::CostFunction *inner, std::vector<Anchor> anchors)
	    : inner(inner), anchors(std::move(anchors))
	{
		assert(this->anchors.size() == this->inner->parameter_block_sizes().size());
		assert(this->anchors.size() <= kMaxParameterBlocks);

		this->set_num_residuals(this->inner->num_residuals());
		*this->mutable_parameter_block_sizes() = this->inner->parameter_block_sizes();
	}

	bool
	Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
	{
		if (jacobians == nullptr) {
			return this->inner->Evaluate(parameters, residuals, nullptr);
		}

		std::array<const double *, kMaxParameterBlocks> anchored{};
		for (size_t i = 0; i < this->anchors.size(); i++) {
			anchored[i] = this->anchors[i].linearization_point != nullptr
			                  ? this->anchors[i].linearization_point
			                  : parameters[i];
		}

		/*
		 * The jacobian pass goes first and its residuals land straight in the output, where the pass below
		 * overwrites them with the ones taken where the states actually are. Saves keeping a scratch buffer in
		 * a function Ceres calls from several threads at once.
		 */
		if (!this->inner->Evaluate(anchored.data(), residuals, jacobians)) {
			return false;
		}

		for (size_t i = 0; i < this->anchors.size(); i++) {
			const Anchor &anchor = this->anchors[i];

			if (anchor.linearization_point == nullptr || anchor.manifold == nullptr ||
			    jacobians[i] == nullptr) {
				continue;
			}

			/*
			 * Ceres turns an ambient jacobian into the tangent one the solver actually uses by multiplying
			 * it by the manifold's plus jacobian *at the point the block currently holds*. An ambient
			 * derivative taken at the anchor comes back expressed in the wrong tangent basis, off by
			 * however far the block has rotated since, which is not a small error and does not show up as
			 * anything except the tracking coming apart.
			 *
			 * Lifting through the anchor's own plus jacobian and back down through the current point's
			 * minus jacobian leaves Ceres holding exactly the tangent jacobian at the anchor, since the
			 * minus jacobian is the left inverse of the plus one at the same point.
			 */
			const int ambient_size = anchor.manifold->AmbientSize();
			const int tangent_size = anchor.manifold->TangentSize();

			SmallRowMajorMatrix plus(ambient_size, tangent_size);
			SmallRowMajorMatrix minus(tangent_size, ambient_size);

			if (!anchor.manifold->PlusJacobian(anchor.linearization_point, plus.data()) ||
			    !anchor.manifold->MinusJacobian(parameters[i], minus.data())) {
				return false;
			}

			Map<RowMajorMatrixXd> jacobian(jacobians[i], this->num_residuals(), ambient_size);
			jacobian = jacobian * (plus * minus);
		}

		return this->inner->Evaluate(parameters, residuals, nullptr);
	}

private:
	//! Big enough for any manifold jacobian in this problem, so the lift never leaves the stack.
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, kMaxAmbientSize, kMaxAmbientSize>
	    SmallRowMajorMatrix;

	std::unique_ptr<ceres::CostFunction> inner;

	std::vector<Anchor> anchors;
};

/*!
 * Names one parameter block in a way that stays valid even after the window has moved past it.
 *
 * The raw `double *` is only good for as long as the build that handed it out. The ring slot it points into is
 * reused by the next keyframe to take that slot, and whether the block is in the problem at all comes out of masks
 * that are recomputed from scratch every build. A marginalization prior outlives both of those and has to be able
 * to ask "is the block I was linearized around still in the solve, and where is it now", which a pointer cannot
 * answer and this can.
 */
struct ParameterBlockRef
{
	enum class Kind
	{
		WorldGravity,
		ImuExtrinsics,
		DeviceKeyframe,
		KeyframeImuBias,
	};

	Kind kind;

	//! Fusion device index. Meaningless for @ref Kind::WorldGravity.
	uint32_t device_idx;

	//! Ring slot, as @ref SensorFusion::keyframeIdxLocked hands out. Only meaningful for the per-keyframe kinds.
	uint32_t ring_slot;

	/*!
	 * Whether this parameter block applies to this device in some way.
	 *
	 * @note It's important to not conflate this with "is this parameter marginalizable away".
	 *       The answer is going to be *no* for @ref Kind::WorldGravity. It can never be marginalized away.
	 */
	bool
	appliesToDevice(uint32_t device_idx) const
	{
		switch (this->kind) {
		case Kind::WorldGravity: {
			return true;
		}
		case Kind::ImuExtrinsics:
		case Kind::DeviceKeyframe:
		case Kind::KeyframeImuBias: {
			return this->device_idx == device_idx;
		}
		}

		assert(!"Unreachable");
		return false;
	}
};

/*!
 * The marginalization prior carried from one solve to the next.
 */
struct MarginalizationPrior
{
	/*!
	 * Whether there is a prior to add to the problem at all.
	 *
	 * False both before one has ever been computed and after one has been discarded for no longer describing
	 * anything the solve still holds, see @ref SensorFusionProblem::dropPriorIfBlocksMissingLocked.
	 */
	bool valid{false};

	//! One retained block, and what it held at the moment the prior was linearized around it.
	struct Block
	{
		/*!
		 * Which block this is.
		 *
		 * Held as a reference rather than a pointer precisely because it has to outlive the build it came
		 * from. @ref ParameterBlocks::resolve turns it back into the pointer Ceres wants.
		 */
		ParameterBlockRef ref;

		/*!
		 * The ambient values the block held when it was marginalized.
		 *
		 * The factor is a statement about how far the block has moved from here, so this has to be a copy: the
		 * storage it came out of is reused by the next keyframe to take that slot.
		 */
		std::vector<double> linearization_point;
	};

	/*!
	 * The retained blocks the prior is a distribution over, in the order its information matrix is laid out in.
	 */
	std::vector<Block> blocks{};

	//! `J^*`, the square root of the marginalized Hessian, laid out over the tangent spaces of @ref blocks.
	MatrixXd j{};

	//! `r^*`, what the factor evaluates to with every block still sitting at its linearization point.
	VectorXd e0{};

	/*!
	 * The residual block the current build added for this prior, or null if the current problem does not hold it.
	 *
	 * Only good for as long as the build that handed it out, which is exactly the window in which the next
	 * marginalization asks whether this factor is among the residuals it is about to fold in.
	 */
	ceres::ResidualBlockId residual{nullptr};
};

/*!
 * The linear factor a marginalization leaves behind, `|J^* dx + r^*|^2`, evaluated against however far the blocks it
 * was linearized around have moved since.
 *
 * @ref marginalize() produces the `J^*` and the `r^*` this carries. The blocks have to be handed over in the order
 * their tangent spaces were laid out in `J^*`'s columns, which is the order @ref MarginalizationPrior::blocks holds
 * them in.
 */
class MarginalizationPriorCostFunction final : public ceres::CostFunction
{
public:
	//! One of the blocks the prior is a distribution over, frozen at what it held when it was linearized.
	struct Block
	{
		//! The ambient values of the block at the moment of marginalization.
		std::vector<double> linearization_point{};

		/*!
		 * The manifold the block sits on, or null for one that lives in a plain Euclidean space.
		 *
		 * Has to stay the manifold the block is added to the problem with: the lift in @ref Evaluate is only
		 * the inverse of that same manifold's plus jacobian.
		 */
		const ceres::Manifold *manifold{nullptr};
	};

	MarginalizationPriorCostFunction(MatrixXd j, VectorXd e0, std::vector<Block> blocks)
	    : j(std::move(j)), e0(std::move(e0)), blocks(std::move(blocks))
	{
		assert(this->j.rows() == this->e0.rows());
		// The states `J^*` was laid out over and the blocks named here have to be the same set.
		assert(totalTangentSize(this->blocks) == this->j.cols());

		this->set_num_residuals(static_cast<int>(this->e0.rows()));

		for (const Block &block : this->blocks) {
			assert(block.manifold == nullptr || block.manifold->AmbientSize() == ambientSize(block));

			this->mutable_parameter_block_sizes()->push_back(ambientSize(block));
		}
	}

	bool
	Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
	{
		// How far each block has moved from its linearization point, stacked in the column order of `J^*`.
		VectorXd delta_chi(this->j.cols());

		int tangent_offset = 0;
		for (size_t i = 0; i < this->blocks.size(); i++) {
			const Block &block = this->blocks[i];
			const int tangent_size = tangentSize(block);

			if (block.manifold != nullptr) {
				if (!block.manifold->Minus(parameters[i], block.linearization_point.data(),
				                           delta_chi.data() + tangent_offset)) {
					return false;
				}
			} else {
				delta_chi.segment(tangent_offset, tangent_size) =
				    Map<const VectorXd>(parameters[i], tangent_size) -
				    Map<const VectorXd>(block.linearization_point.data(), tangent_size);
			}

			tangent_offset += tangent_size;
		}

		Map<VectorXd>(residuals, this->num_residuals()) = this->e0 + (this->j * delta_chi);

		if (jacobians == nullptr) {
			return true;
		}

		tangent_offset = 0;
		for (size_t i = 0; i < this->blocks.size(); i++) {
			const Block &block = this->blocks[i];
			const int tangent_size = tangentSize(block);
			const int block_tangent_offset = tangent_offset;

			tangent_offset += tangent_size;

			if (jacobians[i] == nullptr) {
				continue;
			}

			Map<RowMajorMatrixXd> block_jacobian(jacobians[i], this->num_residuals(), ambientSize(block));

			if (block.manifold == nullptr) {
				block_jacobian = this->j.middleCols(block_tangent_offset, tangent_size);
				continue;
			}

			/*
			 * Ceres asks for the derivative against the ambient parameters and multiplies it by the
			 * manifold's plus jacobian itself. The minus jacobian is the left inverse of the plus one at
			 * the same point, so lifting with it here leaves the solver holding exactly the tangent-space
			 * `J^*` this was linearized as, wherever the block has drifted to since.
			 *
			 * Holding it fixed is the point rather than a shortcut: the exact derivative of `Minus`
			 * against a block that has moved off its linearization point picks up a right-jacobian
			 * term, first order in how far it moved, and a prior that re-linearized itself would stop
			 * being the factor the marginalization solved for. OKVIS and VINS both freeze it the same
			 * way.
			 */
			RowMajorMatrixXd minus_jacobian(tangent_size, ambientSize(block));
			if (!block.manifold->MinusJacobian(parameters[i], minus_jacobian.data())) {
				return false;
			}

			block_jacobian = this->j.middleCols(block_tangent_offset, tangent_size) * minus_jacobian;
		}

		return true;
	}

private:
	static int
	ambientSize(const Block &block)
	{
		return static_cast<int>(block.linearization_point.size());
	}

	static int
	tangentSize(const Block &block)
	{
		return block.manifold != nullptr ? block.manifold->TangentSize() : ambientSize(block);
	}

	static int
	totalTangentSize(const std::vector<Block> &blocks)
	{
		int total = 0;
		for (const Block &block : blocks) {
			total += tangentSize(block);
		}

		return total;
	}

	//! `J^*`, the square root of the marginalized Hessian, laid out over the tangent spaces of @ref blocks.
	MatrixXd j;

	//! `r^*`, what the factor evaluates to with every block still sitting at its linearization point.
	VectorXd e0;

	std::vector<Block> blocks;
};

struct ParameterBlocks
{
	// The world has a single magnitude of gravity (static parameter if kOptimizeWorldGravity)
	SeedableVector<double, WorldGravity<double>::kNumParameters> world_gravity;

	// Each device has static extrinsics over the lifetime of the fusion
	std::array<SeedableVector<double, ImuExtrinsics<double>::kNumParameters>, XRT_CONSTELLATION_MAX_DEVICES>
	    imu_extrinsics;

	std::array<             // Each device has some state through all keyframes in which it was observed
	    std::array<         // Each keyframe an IMU bias
	        SeedableVector< // Each IMU bias has some parameters
	            double,
	            ImuBias<double>::kNumParameters>,
	        kSlidingWindowSize>,
	    XRT_CONSTELLATION_MAX_DEVICES>
	    keyframe_imu_biases;
	std::array<             // Each device has some state through all keyframes in which it was observed
	    std::array<         // Each keyframe has a pose with velocity
	        SeedableVector< // Each pose with velocity has some parameters
	            double,
	            PoseWithVelocity<double>::kNumParameters>,
	        kSlidingWindowSize>,
	    XRT_CONSTELLATION_MAX_DEVICES>
	    device_keyframes;

	//! All the residuals whose oldest keyframe is this one.
	std::array<     // Each device has some data for each keyframe
	    std::array< // Each keyframe has some list of marginalization parameters attached to it
	        std::vector<ceres::ResidualBlockId>,
	        kSlidingWindowSize>,
	    XRT_CONSTELLATION_MAX_DEVICES>
	    per_device_keyframe_marginalization_residuals{};

	/*!
	 * All the parameters that are part of a specific keyframe. These are the parameters that are going to be
	 * marginalized when this keyframe gets dropped.
	 */
	std::array<std::vector<double *>, kSlidingWindowSize> per_keyframe_marginalization_parameters{};

	/*!
	 * Reverse lookup from the pointer Ceres knows a block by to what that block actually is, rebuilt as the
	 * blocks are added each build.
	 *
	 * Ceres hands back raw `double *` when asked what a residual touches, and marginalization has to turn those
	 * back into something that still means anything a build later.
	 */
	std::map<const double *, ParameterBlockRef> block_refs{};

	ParameterBlocks() = default;

	/*!
	 * The live storage for the block @p ref names, or null when that block is not currently seeded and so has
	 * nothing meaningful in it.
	 */
	double *
	resolve(const ParameterBlockRef &ref)
	{
		switch (ref.kind) {
		case ParameterBlockRef::Kind::WorldGravity:
			return this->world_gravity.seeded ? this->world_gravity.vec.data() : nullptr;

		case ParameterBlockRef::Kind::ImuExtrinsics: {
			auto &block = this->imu_extrinsics[ref.device_idx];
			return block.seeded ? block.vec.data() : nullptr;
		}

		case ParameterBlockRef::Kind::DeviceKeyframe: {
			auto &block = this->device_keyframes[ref.device_idx][ref.ring_slot];
			return block.seeded ? block.vec.data() : nullptr;
		}

		case ParameterBlockRef::Kind::KeyframeImuBias: {
			auto &block = this->keyframe_imu_biases[ref.device_idx][ref.ring_slot];
			return block.seeded ? block.vec.data() : nullptr;
		}
		}

		assert(!"Unreachable");
		return nullptr;
	}

	/*!
	 * Freezes the block @p ref names at its current value as the point everything linearizes it at, and hands
	 * that point back.
	 *
	 * Null for a block that is not currently seeded, and for the states that outlive the window.
	 */
	const double *
	anchor(const ParameterBlockRef &ref)
	{
		switch (ref.kind) {
		/*
		 * The states that outlive the window are left to re-linearize where they are, and to be
		 * re-snapshotted as the prior's linearization point at every marginalization.
		 *
		 * Anchoring is only a good trade for a state that stays near its anchor. These are static parameters
		 * seeded once and solved for across the whole session, so a point picked at the first marginalization
		 * is one they spend the rest of the run moving away from, while every factor that touches them would
		 * go on linearizing about it.
		 *
		 * In `Optimization-based VINS: Consistency, Marginalization, and FEJ`
		 * https://pgeneva.com/downloads/papers/Chen2023IROS.pdf
		 * Chen et al. finds that biases and auxiliary variables are safe to re-linearize, "This analytical
		 * nullspace also shows that only orientation, position, velocity, and features affect the
		 * dimensionality, and therefore, there is no need to FEJ the biases or auxiliary variables".
		 */
		case ParameterBlockRef::Kind::WorldGravity:
		case ParameterBlockRef::Kind::ImuExtrinsics:
		case ParameterBlockRef::Kind::KeyframeImuBias: return nullptr;

		case ParameterBlockRef::Kind::DeviceKeyframe:
			return anchorBlock(this->device_keyframes[ref.device_idx][ref.ring_slot]);
		}

		assert(!"Unreachable");
		return nullptr;
	}

private:
	template <typename Block>
	static const double *
	anchorBlock(Block &block)
	{
		if (!block.seeded) {
			return nullptr;
		}

		block.anchor();

		return block.anchorData();
	}
};

/*!
 * The observation-derived world pose of device `device_fusion_idx` at `keyframe`, or `false` if the keyframe holds no
 * observation of it.
 *
 * Where several cameras of the mosaic saw the device, the one with the most blobs wins: its PnP seed is the
 * best-conditioned of the set, and averaging across cameras would need a rotation mean this has no use for elsewhere.
 */
bool
observedDevicePoseLocked(const Keyframe &keyframe, uint32_t device_fusion_idx, xrt_pose &out_Tcv_world_device)
{
	const DeviceObservation *best = nullptr;

	for (uint32_t i = 0; i < keyframe.num_observations; i++) {
		const DeviceObservation &observation = keyframe.observations[i];

		if (observation.device_idx != device_fusion_idx) {
			continue;
		}

		if (best == nullptr || observation.num_points > best->num_points) {
			best = &observation;
		}
	}

	if (best == nullptr) {
		return false;
	}

	math_pose_transform(&best->Tcv_world_cam, &best->Tcv_cam_device_seed, &out_Tcv_world_device);

	return true;
}

/*!
 * Seeds the world-frame velocity of `device_fusion_idx` at keyframe age `keyframe_age`, whose own pose seed is
 * `Tcv_world_device`, by differencing the observation-derived positions of the nearest keyframe on either side that
 * also saw it.
 *
 * Zero is a poor seed: it disagrees with what the IMU preintegrated by the whole of the device's actual motion, and
 * the preintegration factor has to spend its first steps absorbing that before it can say anything about bias. A
 * difference of two PnP seeds is noisy, but it is centred on the right answer, which is all a seed has to be.
 */
Vector3d
seedVelocityLocked(const SensorFusion &fusion,
                   uint32_t device_fusion_idx,
                   uint32_t keyframe_age,
                   const xrt_pose &Tcv_world_device)
{
	/*
	 * Age counts backwards from the newest keyframe, so the newer of any pair is the one with the smaller age.
	 * Both endpoints start collapsed onto the keyframe being seeded, which is what makes a one-sided difference
	 * fall out of the same arithmetic as a central one.
	 */
	const Keyframe &keyframe = fusion.keyframeLocked(keyframe_age);

	xrt_pose Tcv_world_device_newer = Tcv_world_device;
	xrt_pose Tcv_world_device_older = Tcv_world_device;
	timepoint_ns newer_timestamp_ns = keyframe.timestamp_ns;
	timepoint_ns older_timestamp_ns = keyframe.timestamp_ns;
	bool have_neighbour = false;

	for (uint32_t age = keyframe_age; age-- > 0;) {
		const Keyframe &newer = fusion.keyframeLocked(age);

		if (!observedDevicePoseLocked(newer, device_fusion_idx, Tcv_world_device_newer)) {
			continue;
		}

		newer_timestamp_ns = newer.timestamp_ns;
		have_neighbour = true;
		break;
	}

	for (uint32_t age = keyframe_age + 1; age < fusion.num_keyframes; age++) {
		const Keyframe &older = fusion.keyframeLocked(age);

		if (!observedDevicePoseLocked(older, device_fusion_idx, Tcv_world_device_older)) {
			continue;
		}

		older_timestamp_ns = older.timestamp_ns;
		have_neighbour = true;
		break;
	}

	// The only keyframe in the window that sees this device, so there is nothing to difference against.
	if (!have_neighbour) {
		return Vector3d::Zero();
	}

	const time_duration_ns dt_ns = newer_timestamp_ns - older_timestamp_ns;

	/*
	 * Keyframes closer together than the tolerance that groups an exposure are the same instant as far as this is
	 * concerned, and dividing by that interval turns blob noise into an arbitrarily large velocity.
	 */
	if (dt_ns < kKeyframeGroupingToleranceNs) {
		return Vector3d::Zero();
	}

	const Vector3d velocity = (map_vec3(Tcv_world_device_newer.position).cast<double>() -
	                           map_vec3(Tcv_world_device_older.position).cast<double>()) /
	                          (static_cast<double>(dt_ns) * 1e-9);

	/*
	 * One bad PnP seed is enough to produce a velocity that leaves the solve worse off than starting it at rest.
	 * We really don't want devices flying off.
	 */
	if (velocity.norm() > kMaxSeedVelocityMetersPerSecond) {
		return Vector3d::Zero();
	}

	return velocity;
}

/*!
 * The RMS of the problem's residuals in standard deviations, recovered from a Ceres cost.
 *
 * Ceres reports cost as one half of the sum of the squared residuals, so the factor of two undoes that before the
 * mean. Dividing by the residual count is what makes the number comparable between solves: the window holds a
 * different set of keyframes, blobs and devices every tick, and a raw cost moves with all of them.
 */
double
residualRmsSigma(double cost, int num_residuals)
{
	if (num_residuals <= 0) {
		return 0.0;
	}

	return std::sqrt((2.0 * cost) / static_cast<double>(num_residuals));
}

/*!
 * Stops the solve the moment its residuals reach @ref kTargetResidualRmsSigma, rather than letting it spend the rest
 * of the iteration budget polishing a fit that is already inside the noise.
 *
 * The threshold is held as a cost so that the check is a comparison rather than a square root per iteration. Nothing
 * here reads parameter state, which is what lets the solve run without `update_state_every_iteration`.
 */
class ResidualRmsTargetCallback final : public ceres::IterationCallback
{
public:
	ResidualRmsTargetCallback(int num_residuals, double target_rms_sigma)
	    /*
	     * The inverse of `residualRmsSigma`. A problem with no residuals gets a target no cost can meet, so an
	     * empty solve runs its course instead of being reported as an instant success.
	     */
	    : target_cost(num_residuals > 0
	                      ? 0.5 * static_cast<double>(num_residuals) * target_rms_sigma * target_rms_sigma
	                      : -1.0)
	{}

	ceres::CallbackReturnType
	operator()(const ceres::IterationSummary &summary) override
	{
		/*
		 * We always want to compute one full iteration so that it takes into account all cameras, not just the
		 * seed that is "good enough".
		 */
		if (summary.iteration < 2) {
			return ceres::SOLVER_CONTINUE;
		}

		/*
		 * Ceres carries the last accepted cost through an unsuccessful step, so this never mistakes a
		 * rejected trust region step for progress.
		 */
		if (summary.cost <= this->target_cost) {
			return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
		}

		return ceres::SOLVER_CONTINUE;
	}

private:
	double target_cost;
};

extern "C" void *
thread_runner(void *ptr)
{
	SensorFusion *fusion = static_cast<SensorFusion *>(ptr);

	std::vector<constellation::CameraSample> pending;

	// Reserve the maximum amount of samples up front, so we don't keep allocating over time
	pending.reserve(kMaxQueuedCameraSamples);

	os_thread_helper_lock(&fusion->thread);
	while (os_thread_helper_is_running_locked(&fusion->thread)) {
		if (fusion->sample_queue.empty()) {
			os_thread_helper_wait_locked(&fusion->thread);

			// Shutdown wakes us the same way work does, so just exit out if we're no longer supposed to run
			if (!os_thread_helper_is_running_locked(&fusion->thread)) {
				break;
			}
		}

		/*
		 * One swap rather than a pop per sample, and it leaves the queue empty before any of the work below
		 * starts, so a camera thread pushing into it never waits behind the fold.
		 */
		pending.clear();
		// Swap while we have it locked
		std::swap(pending, fusion->sample_queue);

		/*
		 * Camera samples are the only thing a solve is worth running for. A pass with none of them was woken
		 * by pushImuSample() to unpark a preintegration, and only needs to get as far as integrating it.
		 */
		const bool have_new_input = !pending.empty();

		for (const auto &sample : pending) {
			fusion->processCameraSampleLocked(sample);
		}

		if (fusion->tickLocked(have_new_input)) {
			fusion->runFusionLocked();
		}
	}
	os_thread_helper_unlock(&fusion->thread);

	return nullptr;
}

//! Does a single IMU sample integration and updates the passed state parameters
static void
integrateOnce(timepoint_ns &pose_ns,
              const xrt_imu_sample &sample,
              Vector3d &T_world_device,
              Quaterniond &Q_world_device,
              Vector3d &world_velocity,
              const Vector3d &gravity_vec,
              const Vector3d &accel_bias,
              const Vector3d &gyro_bias,
              const Vector3d &accel_scale)
{
	double dt = time_ns_to_s(sample.timestamp_ns - pose_ns);
	// @todo: Handle this properly
	// assert(dt >= 0);
	dt = MAX(0.0, dt);

	Vector3d cur_accel = (map_vec3_f64(sample.accel_m_s2) - accel_bias).cwiseProduct(accel_scale);
	auto cur_gyro = map_vec3_f64(sample.gyro_rad_secs) - gyro_bias;

	auto world_accel = Q_world_device * cur_accel;

	// Cancel out the force of gravity that is being applied
	world_accel += gravity_vec;

	T_world_device += (world_velocity * dt) + (0.5 * world_accel * dt * dt);
	world_velocity += world_accel * dt;

	Q_world_device *= quat_exp_so3(cur_gyro * dt);

	pose_ns = sample.timestamp_ns;
}

static void
integrateToTimestampHistory(const double world_gravity_mag,
                            const timepoint_ns solve_time_ns,
                            const xrt_pose &Tcv_world_device_solve,
                            const xrt_vec3_f64 &solve_world_velocity,
                            const xrt_vec3_f64 &solve_accel_bias,
                            const xrt_vec3_f64 &solve_gyro_bias,
                            const xrt_vec3_f64 &solve_accel_scale,
                            const std::span<const xrt_imu_sample> imu_samples,
                            m_relation_history *relation_history)
{
	assert(imu_samples.size() > 0);

	const Vector3d gravity_vec = {0, world_gravity_mag, 0};

	timepoint_ns pose_ns = solve_time_ns;
	Vector3d T_world_device = map_vec3(Tcv_world_device_solve.position).cast<double>();
	Quaterniond Q_world_device = map_quat(Tcv_world_device_solve.orientation).cast<double>();

	const auto accel_bias = map_vec3_f64(solve_accel_bias);
	const auto gyro_bias = map_vec3_f64(solve_gyro_bias);
	const auto accel_scale = map_vec3_f64(solve_accel_scale);

	Vector3d world_velocity = map_vec3_f64(solve_world_velocity);
	for (const auto &sample : imu_samples) {
		integrateOnce(pose_ns,        //
		              sample,         //
		              T_world_device, //
		              Q_world_device, //
		              world_velocity, //
		              gravity_vec,    //
		              accel_bias,     //
		              gyro_bias,      //
		              accel_scale);   //

		// Normalize the orientation
		Q_world_device.normalize();

		xrt_space_relation relation = {
		    .relation_flags = static_cast<xrt_space_relation_flags>(
		        XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
		        XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT | XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT |
		        XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT),
		    .pose = XRT_POSE_IDENTITY,
		    .linear_velocity = XRT_VEC3_ZERO,
		    .angular_velocity = XRT_VEC3_ZERO,
		};
		map_vec3(relation.pose.position) = T_world_device.cast<float>();
		map_quat(relation.pose.orientation) = Q_world_device.cast<float>();
		map_vec3(relation.linear_velocity) = world_velocity.cast<float>();
		map_vec3(relation.angular_velocity) =
		    (Q_world_device * (map_vec3_f64(sample.gyro_rad_secs) - gyro_bias)).cast<float>();

		m_relation_history_push(relation_history, &relation, pose_ns);
	}
}

static void
integrateToTimestamp(const double world_gravity_mag,
                     const timepoint_ns solve_time_ns,
                     const xrt_pose &Tcv_world_device_solve,
                     const xrt_vec3_f64 &solve_world_velocity,
                     const xrt_vec3_f64 &solve_accel_bias,
                     const xrt_vec3_f64 &solve_gyro_bias,
                     const xrt_vec3_f64 &solve_accel_scale,
                     const std::span<const xrt_imu_sample> imu_samples,
                     const std::optional<xrt_imu_sample> &first_imu_sample_after,
                     xrt_space_relation &relation)
{
	assert(imu_samples.size() > 0);

	Vector3d gravity_vec = {0, world_gravity_mag, 0};

	timepoint_ns pose_ns = solve_time_ns;
	Vector3d T_world_device = map_vec3(Tcv_world_device_solve.position).cast<double>();
	Quaterniond Q_world_device = map_quat(Tcv_world_device_solve.orientation).cast<double>();

	const auto accel_bias = map_vec3_f64(solve_accel_bias);
	const auto gyro_bias = map_vec3_f64(solve_gyro_bias);
	const auto accel_scale = map_vec3_f64(solve_accel_scale);

	xrt_imu_sample end_sample = first_imu_sample_after.value_or(imu_samples.back());

	Vector3d world_velocity = map_vec3_f64(solve_world_velocity);
	for (const auto &sample : imu_samples) {
		integrateOnce(pose_ns,        //
		              sample,         //
		              T_world_device, //
		              Q_world_device, //
		              world_velocity, //
		              gravity_vec,    //
		              accel_bias,     //
		              gyro_bias,      //
		              accel_scale);   //
	}

	// Normalize the orientation
	Q_world_device.normalize();

	relation = {
	    .relation_flags = static_cast<xrt_space_relation_flags>(
	        XRT_SPACE_RELATION_ORIENTATION_VALID_BIT | XRT_SPACE_RELATION_POSITION_VALID_BIT |
	        XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT | XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT |
	        XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT),
	    .pose = XRT_POSE_IDENTITY,
	    .linear_velocity = XRT_VEC3_ZERO,
	    .angular_velocity = XRT_VEC3_ZERO,
	};
	map_vec3(relation.pose.position) = T_world_device.cast<float>();
	map_quat(relation.pose.orientation) = Q_world_device.cast<float>();
	map_vec3(relation.linear_velocity) = world_velocity.cast<float>();
	map_vec3(relation.angular_velocity) =
	    (Q_world_device * (map_vec3_f64(end_sample.gyro_rad_secs) - gyro_bias)).cast<float>();
}

}; // namespace

namespace xrt::tracking::constellation::optimizer::sensor_fusion {

/*
 *
 * SensorFusionProblem
 *
 */

struct SensorFusionProblem
{
	ParameterBlocks parameter_blocks;

	/*!
	 * The camera losses, keyed by the residual count they were sized for. Borrowed by the problem rather than
	 * owned by it, so they have to outlive it.
	 */
	std::map<int, ceres::HuberLoss> camera_losses;

	ceres::Problem problem;

	QuaternionManifold quaternion_manifold;
	PoseManifold pose_manifold;
	PoseWithVelocityManifold pose_with_velocity_manifold;

	/*!
	 * The ring slot each keyframe age mapped to when the snapshot was taken, so that the solved blocks can be
	 * found again after the lock has been dropped. Only meaningful for ages below `snapshot.keyframes.size()`.
	 */
	std::array<uint32_t, kSlidingWindowSize> snapshot_block_idx{};

	//! The marginalization priors for each device.
	std::array<MarginalizationPrior, XRT_CONSTELLATION_MAX_DEVICES> priors{};

	SensorFusionProblem()
	    : parameter_blocks(), quaternion_manifold(), pose_manifold(), pose_with_velocity_manifold()
	{}

	/*!
	 * The ambient size a block of this kind is added with.
	 *
	 * A prior outlives the pointer it was built from, so recovering the size a block later has to go through
	 * what the block *is*. @ref addParameterBlockLocked checks the parameter passes against this.
	 */
	static int
	ambientSizeForKind(ParameterBlockRef::Kind kind)
	{
		switch (kind) {
		case ParameterBlockRef::Kind::WorldGravity: return WorldGravity<double>::kNumParameters;
		case ParameterBlockRef::Kind::ImuExtrinsics: return ImuExtrinsics<double>::kNumParameters;
		case ParameterBlockRef::Kind::DeviceKeyframe: return PoseWithVelocity<double>::kNumParameters;
		case ParameterBlockRef::Kind::KeyframeImuBias: return ImuBias<double>::kNumParameters;
		}

		assert(!"Unreachable");
		return 0;
	}

	//! The manifold a block of this kind is added with, or null for one that lives in a plain Euclidean space.
	ceres::Manifold *
	manifoldForKind(ParameterBlockRef::Kind kind)
	{
		switch (kind) {
		case ParameterBlockRef::Kind::WorldGravity: return nullptr;
		case ParameterBlockRef::Kind::ImuExtrinsics: return &this->quaternion_manifold;
		case ParameterBlockRef::Kind::DeviceKeyframe: return &this->pose_with_velocity_manifold;
		case ParameterBlockRef::Kind::KeyframeImuBias: return nullptr;
		}

		assert(!"Unreachable");
		return nullptr;
	}

	/*!
	 * Wraps @p cost so its jacobians are taken at the anchored linearization points of @p parameter_blocks, or
	 * hands it straight back when first-estimate jacobians are off.
	 *
	 * The blocks have to be given in the order @p cost takes them, which is the order they are handed to
	 * `AddResidualBlock` below.
	 */
	static ceres::CostFunction *
	anchorJacobians(ceres::CostFunction *cost, std::vector<FirstEstimateJacobianCostFunction::Anchor> anchors)
	{
		return new FirstEstimateJacobianCostFunction(cost, std::move(anchors));
	}

	/*!
	 * Adds a parameter block and records what it is, so that marginalization can name it again once the pointer
	 * on its own is no longer valid.
	 *
	 * Every block in the problem goes in through here. A block added behind its back would be one a residual can
	 * reach and @ref ParameterBlocks::block_refs cannot explain, which is exactly the case marginalization has no
	 * way to handle, anyway.
	 */
	void
	addParameterBlockLocked(double *values, int size, ceres::Manifold *manifold, const ParameterBlockRef &ref)
	{
		// A prior rebuilds its factor from the kind alone, so the two have to agree about every block.
		assert(size == ambientSizeForKind(ref.kind));
		assert(manifold == this->manifoldForKind(ref.kind));

		if (manifold != nullptr) {
			this->problem.AddParameterBlock(values, size, manifold);
		} else {
			this->problem.AddParameterBlock(values, size);
		}

		this->parameter_blocks.block_refs.insert_or_assign(values, ref);
	}

	//! Throws the prior away, for the cases where what it was linearized around has stopped being true.
	void
	invalidatePriorLocked(uint32_t fusion_device_idx)
	{
		auto &prior = this->priors[fusion_device_idx];

		prior.valid = false;
		prior.blocks.clear();
		prior.residual = nullptr;
	}

	/*!
	 * Whether the build that is computing @p device_keyframe_masks is going to instantiate the block @p ref
	 * names.
	 *
	 * Mirrors the conditions the parameter passes below actually add blocks under, so the two cannot disagree
	 * about which states exist.
	 */
	static bool
	blockWillBeInstantiatedLocked(const SensorFusion &fusion,
	                              uint32_t device_keyframe_mask,
	                              const ParameterBlockRef &ref)
	{
		switch (ref.kind) {
		case ParameterBlockRef::Kind::WorldGravity:
			// Added unconditionally, every build.
			return true;

		case ParameterBlockRef::Kind::ImuExtrinsics:
			// Added for any device holding at least one state in the window.
			return device_keyframe_mask != 0;

		case ParameterBlockRef::Kind::DeviceKeyframe:
		case ParameterBlockRef::Kind::KeyframeImuBias: {
			uint32_t age = 0;
			if (!fusion.keyframeAgeForIdxLocked(ref.ring_slot, age)) {
				// The keyframe that held this slot has fallen out of the window entirely.
				return false;
			}

			return (device_keyframe_mask & (1u << age)) != 0;
		}
		}

		assert(!"Unreachable");
		return false;
	}

	/*!
	 * Discards the prior unless this build is going to instantiate every block it was linearized around.
	 */
	void
	dropPriorIfBlocksMissingLocked(const SensorFusion &fusion,
	                               const std::array<uint32_t, XRT_CONSTELLATION_MAX_DEVICES> &device_keyframe_masks)
	{
		for (uint32_t fusion_device_idx = 0; fusion_device_idx < fusion.devices.size(); fusion_device_idx++) {
			auto &prior = this->priors[fusion_device_idx];

			if (!prior.valid) {
				continue;
			}

			for (const MarginalizationPrior::Block &block : prior.blocks) {
				const ParameterBlockRef &ref = block.ref;

				assert(ref.appliesToDevice(fusion_device_idx));

				if (blockWillBeInstantiatedLocked(fusion, device_keyframe_masks[fusion_device_idx],
				                                  ref)) {
					continue;
				}

				SF_DEBUG(
				    (&fusion),
				    "Marginalization prior names a block (kind %d, device %u, slot %u) this build does "
				    "not instantiate, dropping the prior.",
				    static_cast<int>(ref.kind), ref.device_idx, ref.ring_slot);

				this->invalidatePriorLocked(fusion_device_idx);
				break;
			}
		}
	}

	/*!
	 * Whether the device's prior is one this build is going to add, and it is a distribution over the block that
	 * @p kind and @p ring_slot name.
	 *
	 * Only meaningful once the prior pass has run, which is why that pass goes first.
	 */
	bool
	priorNamesBlockLocked(uint32_t fusion_device_idx, ParameterBlockRef::Kind kind, uint32_t ring_slot) const
	{
		const auto &prior = this->priors[fusion_device_idx];

		if (!prior.valid) {
			return false;
		}

		for (const MarginalizationPrior::Block &block : prior.blocks) {
			/*
			 * Every block a prior holds already applies to its own device, so the kind and the slot are
			 * what is left to tell them apart.
			 */
			assert(block.ref.appliesToDevice(fusion_device_idx));

			if (block.ref.kind == kind && block.ref.ring_slot == ring_slot) {
				return true;
			}
		}

		return false;
	}

	/*!
	 * The Huber loss for a camera factor carrying `num_residuals` residuals, allocating it on first use.
	 */
	ceres::LossFunction &
	cameraLossForResidualCount(int num_residuals)
	{
		const auto it = this->camera_losses.find(num_residuals);
		if (it != this->camera_losses.end()) {
			return it->second;
		}

		const double dof = static_cast<double>(num_residuals);
		const double threshold = std::sqrt(dof + (kCameraHuberDeltaSigmas * std::sqrt(2.0 * dof)));

		return this->camera_losses.try_emplace(num_residuals, threshold).first->second;
	}

	/*!
	 * Rebuilds the problem from the window. Call with `fusion`'s lock held.
	 *
	 * @return `true` when the problem has something to solve. A window that observes no tracked device produces
	 *         nothing but the constant gravity block, and handing that to Ceres is pure overhead.
	 */
	bool
	buildLocked(SensorFusion &fusion)
	{
		ceres::Problem::Options options{};
		options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
		options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
		options.enable_fast_removal = true;

		this->problem = ceres::Problem(options);

		ParameterBlocks &blocks = this->parameter_blocks;

		/*
		 * Parameters
		 */

		// Clear all the old parameter blocks
		for (auto &parameter_blocks : blocks.per_keyframe_marginalization_parameters) {
			parameter_blocks.clear();
		}

		/*
		 * The pointers these mapped belonged to the problem that was just thrown away. They are refilled as the
		 * blocks go back in below.
		 */
		blocks.block_refs.clear();

		// World gravity
		{
			// Seed before adding as parameter
			if (!blocks.world_gravity.seeded) {
				// @todo Pull the calibrated world gravity from the room setup
				WorldGravity<double>().pack(blocks.world_gravity.seedVec());
			}

			this->addParameterBlockLocked(blocks.world_gravity.data(),          //
			                              WorldGravity<double>::kNumParameters, //
			                              nullptr,                              //
			                              {ParameterBlockRef::Kind::WorldGravity, 0, 0});

			if constexpr (kOptimizeWorldGravity) {
				/*
				 * Gravity varies by a few tenths of a percent over latitude and altitude, so a solve
				 * that wants more than a percent is using it to absorb something else. This keeps it a
				 * free parameter without letting it become a fudge factor.
				 */
				problem.SetParameterLowerBound(blocks.world_gravity.data(),
				                               WorldGravity<double>::kGravityMagIndex, kMinGravity);
				problem.SetParameterUpperBound(blocks.world_gravity.data(),
				                               WorldGravity<double>::kGravityMagIndex, kMaxGravity);
			} else {
				// Stays at whatever it was seeded with, and the IMU factors read it as a constant.
				problem.SetParameterBlockConstant(blocks.world_gravity.data());
			}
		}

		/*
		 * Which keyframe ages hold a state for each device, as a bit per age.
		 *
		 * Nothing but the IMU factors reads a state's velocity, so a state with no preintegration on either
		 * side of it would leave three components sitting in no residual at all and the normal equations rank
		 * deficient. Only the states an IMU factor actually reaches are instantiated, and that has to be worked
		 * out up front because the LED residuals are added in a pass of their own further down and have to
		 * agree with the parameter blocks about which states exist.
		 *
		 * A device with no such states is left out of the solve entirely, biases and extrinsics included. That
		 * covers a device that has been removed, one that is not tracked, one down to its last observed
		 * keyframe on the way to @ref DeviceTrackingState::Lost, and one whose intervals are all inert.
		 */
		static_assert(kSlidingWindowSize <= 32, "the device state masks have a bit per keyframe age");
		std::array<uint32_t, XRT_CONSTELLATION_MAX_DEVICES> device_keyframe_masks{};

		for (uint32_t device_fusion_idx = 0; device_fusion_idx < fusion.devices.size(); device_fusion_idx++) {
			auto &device = fusion.devices[device_fusion_idx];

			if (device.id.load(std::memory_order::relaxed) == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
				continue;
			}

			if (device.tracking_state != DeviceTrackingState::Tracking) {
				continue;
			}

			uint32_t &keyframe_mask = device_keyframe_masks[device_fusion_idx];

			for (uint32_t age = 0; (age + 1) < fusion.num_keyframes; age++) {
				const auto &preintegration = fusion.preintegrationLocked(device, age);

				// Skip preintegrations that have no real data
				if (preintegration.state == DevicePreintegrationState::Dirty ||
				    preintegration.state == DevicePreintegrationState::Empty) {
					continue;
				}

				/*
				 * Both ends of a usable interval observe the device, so this never marks a keyframe
				 * that holds no observation of it.
				 */
				keyframe_mask |= 1u << age;
				keyframe_mask |= 1u << (age + preintegration.relative_start_age);
			}
		}

		/*
		 * The keyframe masks are now final, so we can drop the prior if we're now missing something the prior
		 * *was* referencing.
		 */
		this->dropPriorIfBlocksMissingLocked(fusion, device_keyframe_masks);

		for (uint32_t device_fusion_idx = 0; device_fusion_idx < fusion.devices.size(); device_fusion_idx++) {
			auto &device = fusion.devices[device_fusion_idx];

			const uint32_t keyframe_mask = device_keyframe_masks[device_fusion_idx];

			if (keyframe_mask == 0) {
				// No keyframe saw this device
				continue;
			}

			auto &imu_extrinsics_block = blocks.imu_extrinsics[device_fusion_idx];

			if (!imu_extrinsics_block.seeded) {
				ImuExtrinsics<double>(Quaterniond::Identity()).pack(imu_extrinsics_block.seedVec());
			}

			// IMU extrinsics
			this->addParameterBlockLocked(
			    imu_extrinsics_block.data(),                                     //
			    ImuExtrinsics<double>::kNumParameters,                           //
			    &this->quaternion_manifold,                                      //
			    {ParameterBlockRef::Kind::ImuExtrinsics, device_fusion_idx, 0}); //

			// Add one device keyframe for each state the IMU chain reaches
			auto &device_keyframe_blocks = blocks.device_keyframes[device_fusion_idx];
			for (uint32_t keyframe_age = 0; keyframe_age < fusion.num_keyframes; keyframe_age++) {
				const auto &keyframe = fusion.keyframeLocked(keyframe_age);
				const auto keyframe_idx = fusion.keyframeIdxLocked(keyframe_age);

				auto &keyframe_block = device_keyframe_blocks[keyframe_idx];
				auto &imu_bias_block = blocks.keyframe_imu_biases[device_fusion_idx][keyframe_idx];

				if ((keyframe_mask & (1u << keyframe_age)) == 0) {
					// This keyframe did not see the device
					continue;
				}

				if (!imu_bias_block.seeded) {
					ImuBias<double>(map_vec3_f64(device.latest_accel_bias),
					                map_vec3_f64(device.latest_gyro_bias),
					                map_vec3_f64(device.latest_accel_scale))
					    .pack(imu_bias_block.seedVec());
				}

				// IMU bias for this keyframe
				this->addParameterBlockLocked(                 //
				    imu_bias_block.data(),                     //
				    ImuBias<double>::kNumParameters,           //
				    nullptr,                                   //
				    {ParameterBlockRef::Kind::KeyframeImuBias, //
				     device_fusion_idx,                        //
				     static_cast<uint32_t>(keyframe_idx)});    //
				blocks.per_keyframe_marginalization_parameters[keyframe_idx].push_back(
				    imu_bias_block.data());

				// Set the bounds
				for (int i = 0; i < 3; i++) {
					problem.SetParameterLowerBound(imu_bias_block.data(),
					                               ImuBias<double>::kAccelScaleIndex + i,
					                               kMinAccelScale);
					problem.SetParameterUpperBound(imu_bias_block.data(),
					                               ImuBias<double>::kAccelScaleIndex + i,
					                               kMaxAccelScale);

					problem.SetParameterLowerBound(imu_bias_block.data(),
					                               ImuBias<double>::kAccelBiasIndex + i,
					                               -kMaxAccelBias);
					problem.SetParameterUpperBound(
					    imu_bias_block.data(), ImuBias<double>::kAccelBiasIndex + i, kMaxAccelBias);

					problem.SetParameterLowerBound(
					    imu_bias_block.data(), ImuBias<double>::kGyroBiasIndex + i, -kMaxGyroBias);
					problem.SetParameterUpperBound(
					    imu_bias_block.data(), ImuBias<double>::kGyroBiasIndex + i, kMaxGyroBias);
				}

				if (!keyframe_block.seeded) {
					// The device must have been seen for this to be a valid keyframe
					assert(keyframe.observed_device_mask & (1u << device_fusion_idx));

					xrt_pose Tcv_world_device;
					if (observedDevicePoseLocked(keyframe, device_fusion_idx, Tcv_world_device)) {
						const Vector3d velocity = seedVelocityLocked(
						    fusion, device_fusion_idx, keyframe_age, Tcv_world_device);

						// Seed the pose
						PoseWithVelocity<double>(Tcv_world_device, velocity)
						    .pack(keyframe_block.seedVec());
					}
				}

				// Pose
				this->addParameterBlockLocked(                //
				    keyframe_block.data(),                    //
				    PoseWithVelocity<double>::kNumParameters, //
				    &this->pose_with_velocity_manifold,       //
				    {ParameterBlockRef::Kind::DeviceKeyframe, //
				     device_fusion_idx,                       //
				     static_cast<uint32_t>(keyframe_idx)});   //
				blocks.per_keyframe_marginalization_parameters[keyframe_idx].push_back(
				    keyframe_block.data());
			}
		}

		/*
		 * Residuals
		 */

		// Clear all the old residual blocks
		for (auto &keyframe_residual_blocks : blocks.per_device_keyframe_marginalization_residuals) {
			for (auto &residual_blocks : keyframe_residual_blocks) {
				residual_blocks.clear();
			}
		}

		// Marginalization priors
		for (uint32_t device_fusion_idx = 0; device_fusion_idx < fusion.devices.size(); device_fusion_idx++) {
			auto &prior = this->priors[device_fusion_idx];

			// Whatever id it held belonged to the problem that was just thrown away.
			prior.residual = nullptr;

			if (!prior.valid) {
				continue;
			}

			std::vector<double *> prior_parameters;
			std::vector<MarginalizationPriorCostFunction::Block> prior_blocks;
			prior_parameters.reserve(prior.blocks.size());
			prior_blocks.reserve(prior.blocks.size());

			/*
			 * The oldest keyframe the prior reaches, so that it is folded into the next marginalization at
			 * the same point the residuals it was built from would have been.
			 */
			uint32_t oldest_age = 0;
			bool has_keyframe_block = false;

			for (const MarginalizationPrior::Block &block : prior.blocks) {
				double *values = blocks.resolve(block.ref);

				/*
				 * dropPriorIfBlocksMissingLocked() has already thrown away any prior naming a block
				 * this build does not instantiate, and the parameter passes seed a block before adding
				 * it.
				 */
				assert(values != nullptr);
				if (values == nullptr) {
					this->invalidatePriorLocked(device_fusion_idx);
					break;
				}

				prior_parameters.push_back(values);
				prior_blocks.push_back(
				    {block.linearization_point, this->manifoldForKind(block.ref.kind)});

				const bool per_keyframe = block.ref.kind == ParameterBlockRef::Kind::DeviceKeyframe ||
				                          block.ref.kind == ParameterBlockRef::Kind::KeyframeImuBias;

				uint32_t age = 0;
				if (per_keyframe && fusion.keyframeAgeForIdxLocked(block.ref.ring_slot, age) &&
				    (!has_keyframe_block || age > oldest_age)) {
					// Age counts backwards from the newest keyframe, so the oldest is the largest.
					oldest_age = age;
					has_keyframe_block = true;
				}
			}

			if (!prior.valid) {
				continue;
			}

			prior.residual = problem.AddResidualBlock(
			    new MarginalizationPriorCostFunction(prior.j, prior.e0, std::move(prior_blocks)), //
			    nullptr,                                                                          //
			    prior_parameters);                                                                //

			/*
			 * A prior over nothing but states the window cannot move out from under has no keyframe to be
			 * folded in at. Nothing can invalidate it either, so it rides along until it is replaced.
			 */
			if (has_keyframe_block) {
				blocks
				    .per_device_keyframe_marginalization_residuals[device_fusion_idx]
				                                                  [fusion.keyframeIdxLocked(oldest_age)]
				    .push_back(prior.residual);
			}
		}


		/*
		 * LED residuals, one factor per observation with every LED correspondence of it stacked in, so that
		 * the loss applies to the observation as a whole.
		 */
		for (uint32_t keyframe_age = 0; keyframe_age < fusion.num_keyframes; keyframe_age++) {
			const auto &keyframe = fusion.keyframeLocked(keyframe_age);

			for (uint32_t obs_i = 0; obs_i < keyframe.num_observations; obs_i++) {
				const auto &observation = keyframe.observations[obs_i];
				const auto &camera = fusion.cameras[keyframe.mosaic_idx][observation.camera_idx];

				/*
				 * The solve holds no state for this device at this keyframe to attach the
				 * observation to, so there is nothing for it to constrain. Covers a device that has
				 * been removed, one that is not tracked, and a state the IMU chain does not reach.
				 */
				if ((device_keyframe_masks[observation.device_idx] & (1u << keyframe_age)) == 0) {
					continue;
				}

				const auto keyframe_idx = fusion.keyframeIdxLocked(keyframe_age);

				auto &keyframe_block =
				    parameter_blocks.device_keyframes[observation.device_idx][keyframe_idx];
				auto &device_imu_extrinsics_block =
				    parameter_blocks.imu_extrinsics[observation.device_idx];

				xrt_pose Tcv_cam_world;
				math_pose_invert(&observation.Tcv_world_cam, &Tcv_cam_world);

				const int num_residuals = static_cast<int>(observation.num_points) * kNumLedResiduals;
				ceres::LossFunction &camera_loss = this->cameraLossForResidualCount(num_residuals);

				blocks
				    .per_device_keyframe_marginalization_residuals[observation.device_idx][keyframe_idx]
				    .push_back(problem.AddResidualBlock(             //
				        anchorJacobians(                             //
				            new StaticCameraObservationCostFunction( //
				                new StaticCameraObservationCostFunctor{
				                    camera.params, observation, isometryFromPose(Tcv_cam_world)}, //
				                num_residuals),                                                   //
				            {{keyframe_block.anchorData(), &this->pose_with_velocity_manifold},
				             {device_imu_extrinsics_block.anchorData(), &this->quaternion_manifold}}),
				        &camera_loss,                         //
				        keyframe_block.data(),                //
				        device_imu_extrinsics_block.data())); //
			}
		}

		// IMU preintegration residuals
		for (uint32_t device_fusion_idx = 0; device_fusion_idx < fusion.devices.size(); device_fusion_idx++) {
			auto &device = fusion.devices[device_fusion_idx];

			const uint32_t keyframe_mask = device_keyframe_masks[device_fusion_idx];

			// Nothing of this device is in the solve, so it has no states for its intervals to connect.
			if (keyframe_mask == 0) {
				continue;
			}

			for (uint32_t age = 0; (age + 1) < fusion.num_keyframes; age++) {
				const auto &preintegration = fusion.preintegrationLocked(device, age);

				// Skip preintegrations that have no real data
				if (preintegration.state == DevicePreintegrationState::Dirty ||
				    preintegration.state == DevicePreintegrationState::Empty) {
					continue;
				}

				const uint32_t start_age = age + preintegration.relative_start_age;

				/*
				 * Both ends were instantiated off this same set of intervals, so a factor can never
				 * reach a state the parameter pass did not add, which Ceres would paper over by
				 * adding it itself, without the manifold this block needs.
				 */
				assert((keyframe_mask & (1u << age)) != 0);
				assert((keyframe_mask & (1u << start_age)) != 0);

				const auto start_age_idx = fusion.keyframeIdxLocked(start_age);
				const auto end_age_idx = fusion.keyframeIdxLocked(age);

				auto &start_keyframe_block = blocks.device_keyframes[device_fusion_idx][start_age_idx];
				auto &end_keyframe_block = blocks.device_keyframes[device_fusion_idx][end_age_idx];

				auto &start_imu_bias_block =
				    blocks.keyframe_imu_biases[device_fusion_idx][start_age_idx];
				auto &end_imu_bias_block = blocks.keyframe_imu_biases[device_fusion_idx][end_age_idx];

				// Preintegration residual
				blocks.per_device_keyframe_marginalization_residuals[device_fusion_idx][start_age_idx]
				    .push_back(problem.AddResidualBlock(
				        anchorJacobians( //
				            new ImuCostFunction(
				                new ImuCostFunctor({preintegration.imu_preintegration})), //
				            {{blocks.world_gravity.anchorData(), nullptr},
				             {start_imu_bias_block.anchorData(), nullptr},
				             {start_keyframe_block.anchorData(), &this->pose_with_velocity_manifold},
				             {end_keyframe_block.anchorData(), &this->pose_with_velocity_manifold}}),
				        nullptr,
				        blocks.world_gravity.data(), //
				        // Preintegrate with the IMU bias of the *start* keyframe
				        start_imu_bias_block.data(), //
				        start_keyframe_block.data(), //
				        end_keyframe_block.data())); //

				// IMU bias random walk residual
				blocks.per_device_keyframe_marginalization_residuals[device_fusion_idx][start_age_idx]
				    .push_back(problem.AddResidualBlock(
				        anchorJacobians(
				            new ImuBiasRandomWalkCostFunction(new ImuBiasRandomWalkCostFunctor(
				                {preintegration.imu_preintegration.dt})), //
				            {{start_imu_bias_block.anchorData(), nullptr},
				             {end_imu_bias_block.anchorData(), nullptr}}),
				        nullptr,                     //
				        start_imu_bias_block.data(), //
				        end_imu_bias_block.data())); //
			}

			{ // Oldest keyframe anchor residuals
				const uint32_t oldest_keyframe_age = std::bit_width(keyframe_mask) - 1;

				const auto keyframe_idx = fusion.keyframeIdxLocked(oldest_keyframe_age);

				auto &oldest_bias = blocks.keyframe_imu_biases[device_fusion_idx][keyframe_idx];

				/*
				 * IMU bias anchor residual, without marginalization IMU becomes unconstrained
				 * except by observation, this constrains it.
				 */
				const bool prior_carries_bias = this->priorNamesBlockLocked(
				    device_fusion_idx, ParameterBlockRef::Kind::KeyframeImuBias, keyframe_idx);

				if (!prior_carries_bias) {
					blocks
					    .per_device_keyframe_marginalization_residuals[device_fusion_idx]
					                                                  [keyframe_idx]
					    .push_back(problem.AddResidualBlock(
					        anchorJacobians(
					            new ImuBiasAnchorCostFunction(new ImuBiasAnchorCostFunctor(
					                {ImuBias<double>(map_vec3_f64(device.latest_accel_bias),
					                                 map_vec3_f64(device.latest_gyro_bias),
					                                 map_vec3_f64(device.latest_accel_scale))})),
					            {{oldest_bias.anchorData(), nullptr}}), //
					        nullptr,                                    //
					        oldest_bias.data()));                       //
				}
			}
		}

		/*
		 * Last thing the build does, so the snapshot describes the problem exactly as it is about to be handed
		 * to the solver.
		 */
		this->captureSnapshotLocked(fusion, device_keyframe_masks);

		return this->problem.NumResidualBlocks() > 0;
	}

	/*!
	 * Copies the window out for the logger while the fusion lock is still held.
	 *
	 * The solved half of the snapshot is left alone here, there is nothing solved yet. That part is filled by @ref
	 * fillSolvedSnapshot once Ceres has returned. What this does have to record is the ring slot each keyframe age
	 * maps to, because recovering that later would mean reading `keyframe_head` without the lock.
	 */
	void
	captureSnapshotLocked(SensorFusion &fusion,
	                      const std::array<uint32_t, XRT_CONSTELLATION_MAX_DEVICES> &device_keyframe_masks)
	{
		SolveSnapshot &snapshot = fusion.snapshot;

		snapshot.num_late_samples_dropped = fusion.num_late_samples_dropped;

		for (uint32_t device_fusion_idx = 0; device_fusion_idx < fusion.devices.size(); device_fusion_idx++) {
			const Device &device = fusion.devices[device_fusion_idx];
			SnapshotDevice &out = snapshot.devices[device_fusion_idx];

			out.keyframe_mask = device_keyframe_masks[device_fusion_idx];
			out.in_solve = out.keyframe_mask != 0;
			out.id = device.id.load(std::memory_order_relaxed);
			out.tracking_state = device.tracking_state;
			out.num_observed_keyframes = device.num_observed_keyframes;
			// Overwritten from the solved extrinsics block, and left identity for a device the solve
			// skipped.
			out.Qcv_imu_model = XRT_QUAT_IDENTITY;
		}

		snapshot.num_keyframes = fusion.num_keyframes;
		for (uint32_t age = 0; age < fusion.num_keyframes; age++) {
			const Keyframe &keyframe = fusion.keyframeLocked(age);
			SnapshotKeyframe &out = snapshot.keyframes[age];

			this->snapshot_block_idx[age] = static_cast<uint32_t>(fusion.keyframeIdxLocked(age));

			out.timestamp_ns = keyframe.timestamp_ns;
			out.mosaic_idx = keyframe.mosaic_idx;
			out.sealed = keyframe.sealed;
			out.observed_device_mask = keyframe.observed_device_mask;
			out.devices = {};

			out.num_observations = keyframe.num_observations;
			for (uint32_t i = 0; i < keyframe.num_observations; i++) {
				const DeviceObservation &observation = keyframe.observations[i];
				const CameraDescription &camera =
				    fusion.cameras[keyframe.mosaic_idx][observation.camera_idx];

				SnapshotObservation snapshot_observation{};
				snapshot_observation.mosaic_idx = keyframe.mosaic_idx;
				snapshot_observation.camera_idx = observation.camera_idx;
				snapshot_observation.device_idx = observation.device_idx;
				snapshot_observation.Tcv_world_cam = observation.Tcv_world_cam;
				snapshot_observation.Tcv_cam_device_seed = observation.Tcv_cam_device_seed;
				snapshot_observation.params = camera.params;
				snapshot_observation.num_points = observation.num_points;
				snapshot_observation.points2d = observation.points2d;
				snapshot_observation.points3d = observation.points3d;

				out.observations[i] = snapshot_observation;
			}
		}
	}

	/*!
	 * Reads the solved parameter blocks back into the snapshot.
	 *
	 * Takes no lock and needs none: the blocks belong to this problem, and the ring indices they are reached
	 * through were recorded by @ref captureSnapshotLocked while the lock was held.
	 */
	void
	fillSolvedSnapshot(SolveSnapshot &snapshot)
	{
		const ParameterBlocks &blocks = this->parameter_blocks;

		snapshot.world_gravity_mag =
		    blocks.world_gravity.seeded ? WorldGravity<double>(blocks.world_gravity.vec).gravity_mag : 0.0;

		for (uint32_t device_fusion_idx = 0; device_fusion_idx < snapshot.devices.size(); device_fusion_idx++) {
			SnapshotDevice &device = snapshot.devices[device_fusion_idx];

			if (!device.in_solve) {
				continue;
			}

			const auto &extrinsics_block = blocks.imu_extrinsics[device_fusion_idx];
			if (extrinsics_block.seeded) {
				map_quat(device.Qcv_imu_model) =
				    ImuExtrinsics<double>(extrinsics_block.vec).Q_imu_model.coeffs().cast<float>();
			}

			for (uint32_t age = 0; age < snapshot.keyframes.size(); age++) {
				SolvedDeviceKeyframe &out = snapshot.keyframes[age].devices[device_fusion_idx];

				if ((device.keyframe_mask & (1u << age)) == 0) {
					continue;
				}

				const uint32_t block_idx = this->snapshot_block_idx[age];
				const auto &pose_block = blocks.device_keyframes[device_fusion_idx][block_idx];
				const auto &bias_block = blocks.keyframe_imu_biases[device_fusion_idx][block_idx];

				/*
				 * The mask says the solve reached this state, so both blocks were seeded on the way in.
				 * Checking anyway keeps a logging path from being the thing that trips an assert.
				 */
				if (!pose_block.seeded || !bias_block.seeded) {
					continue;
				}

				const PoseWithVelocity<double> pose{pose_block.vec};
				const ImuBias<double> bias{bias_block.vec};

				out.in_solve = true;
				out.Tcv_world_device = pose.toXrtPose();
				map_vec3_f64(out.velocity) = pose.velocity;
				map_vec3_f64(out.accel_bias) = bias.accel_bias;
				map_vec3_f64(out.gyro_bias) = bias.gyro_bias;
				map_vec3_f64(out.accel_scale) = bias.accel_scale;
			}
		}
	}

	/*!
	 * Drops everything seeded from a device's calibration, so the next build re-seeds it.
	 *
	 * The keyframe poses are deliberately left alone: they are seeded from observations, which a change of IMU
	 * intrinsics says nothing about, and throwing away a converged pose would cost the solve its warm start.
	 */
	void
	resetDeviceCalibrationSeedsLocked(uint32_t device_fusion_idx)
	{
		/*
		 * The prior was linearized against the biases this is throwing out, and there is no correcting it
		 * across a change that large.
		 */
		this->invalidatePriorLocked(device_fusion_idx);

		this->parameter_blocks.imu_extrinsics[device_fusion_idx].unseed();

		for (auto &bias : this->parameter_blocks.keyframe_imu_biases[device_fusion_idx]) {
			bias.unseed();
		}
	}

	void
	evictLocked(SensorFusion &fusion, uint32_t age)
	{
		/*
		 * No lock of its own, and none needed: eviction reaches this from the fusion thread's fold pass, which
		 * is the same thread that builds and solves. Nothing else can be looking at the problem.
		 */
		uint32_t idx = fusion.keyframeIdxLocked(age);

		for (uint32_t fusion_device_idx = 0; fusion_device_idx < fusion.devices.size(); fusion_device_idx++) {
			const auto &marginalized_residuals =
			    this->parameter_blocks
			        .per_device_keyframe_marginalization_residuals[fusion_device_idx][idx];
			const auto &full_marginalized_parameters_vec =
			    this->parameter_blocks.per_keyframe_marginalization_parameters[idx];

			//! The base parameters of this marginalization (M), only the ones that apply to this device.
			std::set<double *> base_parameters;

			for (const auto marginalized_parameter : full_marginalized_parameters_vec) {
				// Skip constant parameter blocks
				if (this->problem.IsParameterBlockConstant(marginalized_parameter)) {
					continue;
				}

				// Skip parameters that do not apply to this device.
				if (!this->parameter_blocks.block_refs.at(marginalized_parameter)
				         .appliesToDevice(fusion_device_idx)) {
					continue;
				}

				base_parameters.emplace(marginalized_parameter);
			}

			if (base_parameters.empty()) {
				/*
				 * Nothing of this device lived at the evicted keyframe, so there is nothing to
				 * marginalize and the prior it already holds is still exactly as good as it was.
				 */
				continue;
			}

			std::set<double *> connected_parameters_set;
			for (const auto marginalization_residual : marginalized_residuals) {
				std::vector<double *> connected_parameters_vec;
				this->problem.GetParameterBlocksForResidualBlock(marginalization_residual,
				                                                 &connected_parameters_vec);

				for (const auto connected_parameter : connected_parameters_vec) {
					// This parameter is a base parameter.
					if (base_parameters.count(connected_parameter) != 0) {
						continue;
					}

					// Skip constant parameter blocks
					if (this->problem.IsParameterBlockConstant(connected_parameter)) {
						continue;
					}

					connected_parameters_set.emplace(connected_parameter);
				}
			}

			if (connected_parameters_set.empty()) {
				/*
				 * Everything those residuals touched is being marginalized away, so there is no state
				 * left for a factor to be a statement about.
				 */
				this->invalidatePriorLocked(fusion_device_idx);
				continue;
			}

			ceres::Problem::EvaluateOptions options;
			options.apply_loss_function = true;
			options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
			options.parameter_blocks.insert(options.parameter_blocks.end(), base_parameters.begin(),
			                                base_parameters.end());
			options.parameter_blocks.insert(options.parameter_blocks.end(),
			                                connected_parameters_set.begin(),
			                                connected_parameters_set.end());
			options.residual_blocks = marginalized_residuals;

			std::vector<double> residual_vec;
			ceres::CRSMatrix jacobian_crs;
			this->problem.Evaluate(options, nullptr, &residual_vec, nullptr, &jacobian_crs);

			auto residuals =
			    Eigen::Map<VectorXd, Eigen::Unaligned>(residual_vec.data(), residual_vec.size());

			MatrixXd jacobian;
			matrixCrsToEigen(jacobian_crs, jacobian);

			/*
			 * The Jacobian's columns are the tangent sizes of the parameter blocks, in the order
			 * they were put into options.parameter_blocks, so the base/remainder split has to be
			 * counted in tangent dimensions rather than in parameter blocks.
			 */
			int num_base_states = 0;
			for (double *base_parameter : base_parameters) {
				num_base_states += this->problem.ParameterBlockTangentSize(base_parameter);
			}

			VectorXd e0_marg;
			MatrixXd j_marg;
			marginalize(residuals, jacobian, num_base_states, e0_marg, j_marg);

			/*
			 * Whether the factor this device already carries is one of the residuals being folded in here.
			 *
			 * When it is not, everything that prior knew goes on the floor: the new factor is built from
			 * the residuals at this keyframe alone and then overwrites it. That should not happen while the
			 * device's oldest state is the one the prior was registered against, which is why it is worth
			 * hearing about when it does.
			 */
			const bool had_prior = this->priors[fusion_device_idx].valid;
			const ceres::ResidualBlockId previous_residual = this->priors[fusion_device_idx].residual;
			const bool folded_in = had_prior && previous_residual != nullptr &&
			                       std::find(marginalized_residuals.begin(), marginalized_residuals.end(),
			                                 previous_residual) != marginalized_residuals.end();

			/*
			 * Replaced wholesale: whatever this device carried before is either folded into the factor
			 * being built here, or was dropped along with the residuals that are going away.
			 */
			this->invalidatePriorLocked(fusion_device_idx);
			auto &prior = this->priors[fusion_device_idx];

			bool blocks_resolved = true;
			for (double *connected_parameter : connected_parameters_set) {
				const auto it = this->parameter_blocks.block_refs.find(connected_parameter);

				/*
				 * Every block in the problem went in through addParameterBlockLocked(), so a residual
				 * cannot reach one that is not in here.
				 */
				assert(it != this->parameter_blocks.block_refs.end());
				if (it == this->parameter_blocks.block_refs.end()) {
					blocks_resolved = false;
					break;
				}

				const ParameterBlockRef &ref = it->second;

				/*
				 * Anchoring the state here is what keeps this factor and the residuals that outlive it
				 * talking about the same linearization, and a state already anchored by an earlier
				 * marginalization keeps the point it was anchored at rather than being walked forward.
				 *
				 * Without first-estimate jacobians there is no anchor, and the factor is linearized
				 * around wherever the solve has just left the state.
				 */
				const double *anchored = this->parameter_blocks.anchor(ref);
				const double *linearization_point =
				    anchored != nullptr ? anchored : connected_parameter;

				prior.blocks.push_back(
				    {ref, {linearization_point, linearization_point + ambientSizeForKind(ref.kind)}});
			}

			if (!blocks_resolved) {
				this->invalidatePriorLocked(fusion_device_idx);
				continue;
			}

			prior.j = std::move(j_marg);
			prior.e0 = std::move(e0_marg);
			prior.valid = true;

			/*
			 * `r^*` is what the new factor evaluates to with every block still where it is right now, so a
			 * healthy one starts out near zero: the states it is a distribution over are the ones the solve
			 * just settled on. A large norm means the marginalized residuals disagreed with the solved
			 * estimate, and the prior is about to pull the window towards something it was never at.
			 */
			SF_DEBUG((&fusion), "Marginalized keyframe %u of device %u into a %u state prior, |r*| %f, %s.",
			         age, fusion_device_idx, static_cast<uint32_t>(prior.e0.rows()), prior.e0.norm(),
			         !had_prior ? "first one it has held"
			                    : (folded_in ? "folding in the one it held" : "replacing the one it held"));

			if (had_prior && !folded_in) {
				SF_WARN(
				    (&fusion),
				    "Device %u's previous marginalization prior was not among the residuals folded in "
				    "here, so everything it carried is lost.",
				    fusion_device_idx);
			}
		}

		/*
		 * Mark as needing seeding for all devices. The slot is about to be handed to a keyframe of its own, so
		 * whatever the state living here was anchored at stops meaning anything.
		 */
		for (auto &keyframes : this->parameter_blocks.device_keyframes) {
			keyframes[idx].unseed();
		}

		for (auto &keyframes : this->parameter_blocks.keyframe_imu_biases) {
			keyframes[idx].unseed();
		}
	}
};

/*
 *
 * Device
 *
 */

Device::Device()
    : id(XRT_CONSTELLATION_INVALID_DEVICE_ID), //
      calibrated_accel_bias({0.0, 0.0, 0.0}),  //
      calibrated_gyro_bias({0.0, 0.0, 0.0}),   //
      calibrated_accel_scale({1.0, 1.0, 1.0}), //
      imu_write_count(0),                      //
      imu_samples()                            //
{
	/*
	 * @todo: The "calibrated" bias parameters need to be filled in from some other component which has them.
	 *        On CV1 the headset is calibrated from the factory, so this is fine *for now*, but this will need to be
	 *        done properly in the future.
	 */

	m_relation_history_create(&this->relation_history);

	this->reset();
}

Device::~Device()
{
	m_relation_history_destroy(&this->relation_history);
}

void
Device::reset()
{
	this->led_model = nullptr;
	this->max_dead_reckoning_ns = 0;

	/*
	 * The last solve belonged to whichever device held this slot before. Left alone, a device that is registered
	 * into a used slot (a controller power-cycling, say) would be handed the previous occupant's pose to
	 * dead-reckon from until its own first solve, and the constructor would leave these uninitialized.
	 */
	this->has_latest_solve = false;
	this->latest_solve_time_ns = 0;
	this->Tcv_world_device_latest = XRT_POSE_IDENTITY;
	this->latest_world_velocity = {0.0, 0.0, 0.0};

	m_relation_history_clear(this->relation_history);

	/*
	 * The calibration belonged to whichever device held this slot last, so it goes back to "uncalibrated" rather
	 * than being handed to the next one. `setDeviceImuCalibration` is what fills it in again.
	 */
	this->calibrated_accel_bias = {0.0, 0.0, 0.0};
	this->calibrated_gyro_bias = {0.0, 0.0, 0.0};
	this->calibrated_accel_scale = {1.0, 1.0, 1.0};
	this->Qcv_imu_model = XRT_QUAT_IDENTITY;

	this->latest_accel_bias = this->calibrated_accel_bias;
	this->latest_gyro_bias = this->calibrated_gyro_bias;
	this->latest_accel_scale = this->calibrated_accel_scale;

	this->tracking_state = DeviceTrackingState::Uninitialized;
	this->num_observed_keyframes = 0;

	for (auto &preintegration : this->preintegrations) {
		preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Dirty);
	}

	/*
	 * The ring belongs to this slot rather than to whatever device held it last. Nothing is ever read at or past
	 * the count, so zeroing that is what makes any stale contents unreachable without having to wipe 280KB.
	 */
	this->imu_write_count.store(0, std::memory_order_relaxed);
}

/*
 *
 * Helper functions
 *
 */

size_t
SensorFusion::keyframeIdxLocked(uint32_t age) const
{
	return (this->keyframe_head - 1 - age) % kSlidingWindowSize;
}

bool
SensorFusion::keyframeAgeForIdxLocked(size_t idx, uint32_t &out_age) const
{
	// Bounded by the window, so a scan is easier than the real math to reason about and is cheap anyway.
	for (uint32_t age = 0; age < this->num_keyframes; age++) {
		if (this->keyframeIdxLocked(age) == idx) {
			out_age = age;
			return true;
		}
	}

	return false;
}

Keyframe &
SensorFusion::keyframeLocked(uint32_t age)
{
	assert(age < this->num_keyframes);

	return this->keyframe_storage[this->keyframeIdxLocked(age)];
}

const Keyframe &
SensorFusion::keyframeLocked(uint32_t age) const
{
	assert(age < this->num_keyframes);

	return this->keyframe_storage[this->keyframeIdxLocked(age)];
}

DevicePreintegration &
SensorFusion::preintegrationLocked(Device &device, uint32_t age)
{
	assert(age < this->num_keyframes);

	return device.preintegrations[this->keyframeIdxLocked(age)];
}

void
SensorFusion::dirtyPreintegrationsForNewObservationLocked(uint32_t device_fusion_idx, uint32_t keyframe_age)
{
	Device &device = this->devices[device_fusion_idx];
	const uint32_t device_bit = 1u << device_fusion_idx;

	// The interval ending here was integrated with nothing to terminate on, or not at all.
	this->preintegrationLocked(device, keyframe_age).state = DevicePreintegrationState::Dirty;

	/*
	 * Whichever interval integrated straight over this keyframe now has a state sitting in the middle of it, so it
	 * has to be cut in two. That is the one ending at the nearest younger keyframe observing this device.
	 */
	for (uint32_t age = keyframe_age; age-- > 0;) {
		if ((this->keyframeLocked(age).observed_device_mask & device_bit) == 0) {
			continue;
		}

		this->preintegrationLocked(device, age).state = DevicePreintegrationState::Dirty;
		break;
	}
}

Keyframe &
SensorFusion::pushKeyframeLocked(const constellation::CameraSample &sample)
{
	/*
	 * The slot about to be reused holds the keyframe falling out of the window, so take its devices off the books
	 * before it is overwritten and the observation counts drift out of step with what the ring actually holds.
	 */
	if (this->num_keyframes == kSlidingWindowSize) {
		uint32_t evicted_age = kSlidingWindowSize - 1;

		const Keyframe &evicted = this->keyframeLocked(evicted_age);

		/*
		 * Marginalization may only ever fold in sealed keyframes, and this is the last moment one is
		 * reachable. `kKeyframeSealLagKeyframes < kSlidingWindowSize` is what guarantees it.
		 */
		assert(evicted.sealed);

		// Evict the age for the fusion problem
		this->problem->evictLocked(*this, evicted_age);

		for (uint32_t i = 0; i < this->devices.size(); i++) {
			if ((evicted.observed_device_mask & (1u << i)) == 0) {
				continue;
			}

			assert(this->devices[i].num_observed_keyframes > 0);
			this->devices[i].num_observed_keyframes -= 1;
		}

		/*
		 * An interval can start at the keyframe being evicted, and the state it integrates from is about to
		 * stop existing. The oldest keyframe's own slot is skipped because it is the one being reused below.
		 */
		for (auto &device : this->devices) {
			for (uint32_t age = 0; age + 1 < this->num_keyframes; age++) {
				DevicePreintegration &preintegration = this->preintegrationLocked(device, age);

				if (age + preintegration.relative_start_age == this->num_keyframes - 1) {
					preintegration.state = DevicePreintegrationState::Dirty;
				}
			}
		}
	} else {
		this->num_keyframes += 1;
	}

	const uint64_t slot = this->keyframe_head % kSlidingWindowSize;
	this->keyframe_head += 1;

	Keyframe &keyframe = this->keyframe_storage[slot];
	keyframe.mosaic_idx = sample.mosaic_index;
	keyframe.timestamp_ns = sample.timestamp_ns;
	keyframe.contributed_camera_mask = 0;
	keyframe.observed_device_mask = 0;
	keyframe.sealed = false;
	/*
	 * Deliberately not clearing `observations`, would be very expensive.
	 */
	keyframe.num_observations = 0;

	/*
	 * Nothing has integrated the interval ending at this keyframe yet, and the slot still holds the preintegration
	 * belonging to the keyframe that just fell out of it.
	 */
	for (auto &device : this->devices) {
		device.preintegrations[slot] = DevicePreintegration::Identity(DevicePreintegrationState::Dirty);
	}

	this->sealAgedKeyframesLocked();

	return keyframe;
}

void
SensorFusion::sealAgedKeyframesLocked()
{
	/*
	 * Walking back from the lag point and stopping at the first sealed keyframe would cause keyframes to
	 * seal out of order. A complete one seals the moment its last camera reports, so the scan has to cover the
	 * whole tail.
	 */
	for (uint32_t age = kKeyframeSealLagKeyframes; age < this->num_keyframes; age++) {
		Keyframe &keyframe = this->keyframeLocked(age);

		if (keyframe.sealed) {
			continue;
		}

		SF_DEBUG(this, "Sealing keyframe %u at %" PRIi64 " on age, cameras 0x%x of 0x%x reported.",
		         keyframe.mosaic_idx, keyframe.timestamp_ns, keyframe.contributed_camera_mask,
		         this->mosaic_camera_masks[keyframe.mosaic_idx]);

		keyframe.sealed = true;
	}
}

void
SensorFusion::updateDeviceLifecycleLocked(Device &device)
{
	const t_constellation_device_id_t device_id = device.id.load(std::memory_order_relaxed);

	if (device_id == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
		return;
	}

	switch (device.tracking_state) {
	case DeviceTrackingState::Uninitialized:
		if (device.num_observed_keyframes >= kMinObservedKeyframesToTrack) {
			SF_INFO(this, "Device %d is tracked, %u keyframes of the window observe it.", device_id,
			        device.num_observed_keyframes);

			device.tracking_state = DeviceTrackingState::Tracking;
		}
		break;

	case DeviceTrackingState::Tracking:
		/*
		 * The window is the entire set of observations the solve has to work with, so once the last one has
		 * aged out of it there is nothing left holding this device's states down and everything the IMU chain
		 * has to say across the gap is dead reckoning.
		 */
		if (device.num_observed_keyframes == 0) {
			SF_INFO(this, "Device %d lost, no keyframe in the window observes it.", device_id);

			device.tracking_state = DeviceTrackingState::Lost;
		}
		break;

	case DeviceTrackingState::Lost:
		/*
		 * Held until tickLocked() has acknowledged it. Going straight back to Tracking from here would skip
		 * tearing down everything the blind stretch invalidated.
		 */
		break;
	}
}

void
SensorFusion::markDirtyLocked()
{
	this->dirty = true;
	os_thread_helper_signal_locked(&this->thread);
}

bool
SensorFusion::buildFusionLocked()
{
	const bool has_something_to_solve = this->problem->buildLocked(*this);

	SF_TRACE(this, "Built sensor fusion optimization problem, %s to solve",
	         has_something_to_solve ? "something" : "nothing");

	return has_something_to_solve;
}

void
SensorFusion::runFusionLocked()
{
	ceres::Problem &problem = this->problem->problem;

	/*
	 * Fixed for the life of this solve since this thread is the only ones to add residuals and we aren't doing that
	 * here.
	 */
	const int num_residuals = problem.NumResiduals();

	ceres::Solver::Options options{};
	options.max_num_iterations = 15;
	/*
	 * The problem is block sparse: a keyframe only touches its own camera observations and the two IMU factors
	 * either side of it.
	 */
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
	// Eigen's sparse backend is the one this Ceres is built with, see CERES_USE_EIGEN_SPARSE.
	options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE;
	options.minimizer_progress_to_stdout = false;
	options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
	options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
	// @todo Tune the tolerances to actual numbers.

	// Ceres does not take ownership, and the callback only has to outlive the `Solve` below.
	ResidualRmsTargetCallback rms_target{num_residuals, kTargetResidualRmsSigma};

	if constexpr (kExitOnRmsFloor) {
		options.callbacks.push_back(&rms_target);
		/*
		 * The callback reads nothing but the cost Ceres has already computed, so the solve is spared pushing
		 * parameter state back out to the blocks every iteration.
		 */
		options.update_state_every_iteration = false;
	}

	ceres::Solver::Summary summary;

	/*
	 * Unlock the fusion thread, since we've built our Ceres problem and nothing else will touch it or the parameter
	 * pointers.
	 */
	os_thread_helper_unlock(&this->thread);
	ceres::Solve(options, &problem, &summary);
	os_thread_helper_lock(&this->thread);

	SolveSnapshot &snapshot = this->snapshot;
	this->problem->fillSolvedSnapshot(snapshot);

#ifdef XRT_FEATURE_RERUN
	if (this->rerun_context) {
		/*
		 * Logging a solve means serializing the whole window out to the stream, which is far too long to
		 * hold the queue shut for. Nothing here is reachable from another thread, so it does not need to be
		 * held for.
		 */
		os_thread_helper_unlock(&this->thread);

		snapshot.num_residuals = num_residuals;
		snapshot.num_parameter_blocks = problem.NumParameterBlocks();
		snapshot.num_iterations = static_cast<uint32_t>(summary.iterations.size());
		snapshot.initial_rms_sigma = residualRmsSigma(summary.initial_cost, num_residuals);
		snapshot.final_rms_sigma = residualRmsSigma(summary.final_cost, num_residuals);
		snapshot.solve_time_s = summary.total_time_in_seconds;
		snapshot.termination = ceres::TerminationTypeToString(summary.termination_type);

		this->rerun_context->logSolve(snapshot);

		/*
		 * Numbered after the fact, so the first solve is 0 and the logger's sequence timeline lines up with
		 * how many solves have actually been logged.
		 */
		snapshot.solve_index += 1;

		os_thread_helper_lock(&this->thread);
	}
#endif

	std::array<bool, XRT_CONSTELLATION_MAX_DEVICES> device_updated = {};
	for (uint32_t i = 0; i < snapshot.num_keyframes; i++) {
		const auto &keyframe = snapshot.keyframes[i];

		for (uint32_t fusion_device_idx = 0; fusion_device_idx < keyframe.devices.size(); fusion_device_idx++) {
			const auto &device = keyframe.devices[fusion_device_idx];

			if (!device.in_solve || device_updated[fusion_device_idx]) {
				continue;
			}

			auto &fusion_device = this->devices[fusion_device_idx];

			// we have a previous solve and it's older than the new one
			if (fusion_device.has_latest_solve &&
			    keyframe.timestamp_ns > fusion_device.latest_solve_time_ns) {
				std::array<xrt_imu_sample, kImuBufferSize> samples_buf;
				uint32_t num_samples = 0;
				xrt_imu_sample first_after;

				// Get the IMU samples
				const ImuCollectResult result =
				    collectImuSamplesForRange(fusion_device,                      //
				                              fusion_device.latest_solve_time_ns, //
				                              keyframe.timestamp_ns,              //
				                              samples_buf,                        //
				                              num_samples,                        //
				                              first_after);                       //

				/*
				 * If we got it OK, or there's no sample *after*, that's fine, we're integrating
				 * until the last IMU sample *before* the stop time.
				 */
				if ((result == ImuCollectResult::Ok || result == ImuCollectResult::NoSampleAfterEnd) //
				    && num_samples > 0) {
					integrateToTimestampHistory(
					    this->snapshot.world_gravity_mag,                                 //
					    fusion_device.latest_solve_time_ns,                               //
					    fusion_device.Tcv_world_device_latest,                            //
					    fusion_device.latest_world_velocity,                              //
					    fusion_device.latest_accel_bias,                                  //
					    fusion_device.latest_gyro_bias,                                   //
					    fusion_device.latest_accel_scale,                                 //
					    std::span<const xrt_imu_sample>(samples_buf.data(), num_samples), //
					    fusion_device.relation_history);                                  //

					SF_DEBUG(this,
					         "Filled relation history for device %d from %" PRIi64 " to %" PRIi64,
					         fusion_device.id.load(std::memory_order_acquire),
					         fusion_device.latest_solve_time_ns, keyframe.timestamp_ns);
				}
			}

			fusion_device.has_latest_solve = true;
			fusion_device.latest_solve_time_ns = keyframe.timestamp_ns;
			fusion_device.Tcv_world_device_latest = device.Tcv_world_device;
			fusion_device.latest_world_velocity = device.velocity;

			fusion_device.latest_accel_bias = device.accel_bias;
			fusion_device.latest_gyro_bias = device.gyro_bias;
			fusion_device.latest_accel_scale = device.accel_scale;

			fusion_device.Qcv_imu_model = snapshot.devices[fusion_device_idx].Qcv_imu_model;

			device_updated[fusion_device_idx] = true;

			SF_DEBUG(this, "Got new solved pose for device %d at %" PRIi64,
			         fusion_device.id.load(std::memory_order_acquire), keyframe.timestamp_ns);
		}
	}

	SF_DEBUG(this, "Solved system in %zu iterations, %.3f -> %.3f sigma RMS over %d residuals (%s). Report: %s",
	         summary.iterations.size(),                                //
	         residualRmsSigma(summary.initial_cost, num_residuals),    //
	         residualRmsSigma(summary.final_cost, num_residuals),      //
	         num_residuals,                                            //
	         ceres::TerminationTypeToString(summary.termination_type), //
	         summary.FullReport().c_str());                            //
}

/*
 *
 * "Exported" functions
 *
 */

SensorFusion::SensorFusion()
    : log_level(debug_get_log_option_sensor_fusion_log()), //
      thread(),                                            //
      keyframe_storage(),                                  //
      keyframe_head(0),                                    //
      num_keyframes(0),                                    //
      cameras(),                                           //
      mosaic_camera_masks(),                               //
      devices(),                                           //
      num_devices(0),                                      //
      sample_queue(),                                      //
      num_queued_samples_dropped(0),                       //
      dirty(false),                                        //
      waiting_on_imu(false),                               //
      num_late_samples_dropped(0)                          //
{
	this->problem = new SensorFusionProblem();

	if (debug_get_bool_option_sensor_fusion_enable_rerun()) {
#ifdef XRT_FEATURE_RERUN
		this->rerun_context = std::make_unique<RerunContext>();
		SF_INFO(this, "Sensor fusion Rerun stream enabled");

		if (debug_get_bool_option_sensor_fusion_rerun_spawn()) {
			this->rerun_context->spawnViewer();
		} else {
			this->rerun_context->stream->connect_grpc().exit_on_failure();
		}
#else
		SF_ERROR(this, "Rerun stream requested but XRT_FEATURE_RERUN is not enabled");
#endif
	}

	// Everything the fusion thread touches has to exist before it is let loose on it.
	os_thread_helper_init(&this->thread);
	os_thread_helper_start(&this->thread, thread_runner, this);
}

SensorFusion::~SensorFusion()
{
	// Joins the fusion thread, so nothing can be mid-solve by the time the problem and the stream go away.
	os_thread_helper_destroy(&this->thread);

	delete this->problem;
}

void
SensorFusion::addCamera(const CameraDescription &camera_description)
{
	os_thread_helper_lock(&this->thread);

	// Copy the camera description in
	this->cameras[camera_description.mosaic_idx][camera_description.camera_idx] = camera_description;

	// Sealing a keyframe as soon as its exposure is complete hinges on knowing which cameras owe it a sample.
	this->mosaic_camera_masks[camera_description.mosaic_idx] |= 1u << camera_description.camera_idx;

	os_thread_helper_unlock(&this->thread);
}

void
SensorFusion::addDevice(t_constellation_device_id_t id,
                        const t_constellation_tracker_led_model *led_model,
                        int64_t max_dead_reckoning_ns)
{
	assert(id != XRT_CONSTELLATION_INVALID_DEVICE_ID);

	os_thread_helper_lock(&this->thread);

	// One pass, because the whole list has to be checked for a duplicate before a free slot can be claimed.
	int32_t free_idx = -1;
	for (uint32_t i = 0; i < ARRAY_SIZE(this->devices); i++) {
		const t_constellation_device_id_t slot_id = this->devices[i].id.load(std::memory_order_relaxed);

		if (slot_id == id) {
			// Already added
			os_thread_helper_unlock(&this->thread);
			return;
		}

		if (slot_id == XRT_CONSTELLATION_INVALID_DEVICE_ID && free_idx < 0) {
			free_idx = static_cast<int32_t>(i);
		}
	}

	assert(this->num_devices < XRT_CONSTELLATION_MAX_DEVICES);

	if (free_idx < 0) {
		os_thread_helper_unlock(&this->thread);

		assert(!"Unreachable"); // We failed to find a device slot?

		// Fallback for release mode
		throw std::runtime_error("unreachable: Failed to find device slot for adding");
	}

	Device &device = this->devices[free_idx];

	// Set the slot up before anything can find it.
	device.reset();
	device.led_model = led_model;
	device.max_dead_reckoning_ns = max_dead_reckoning_ns;

	/*
	 * Release, pairing with the acquire in pushImuSample(): a thread that finds this ID is guaranteed to see a slot
	 * that is fully set up rather than one halfway through being written.
	 *
	 * An IMU push that read the *previous* occupant's ID out of this slot can still be in flight here and land one
	 * stale sample in the ring. That is harmless: preintegration for a device only starts once the window observes
	 * it, which needs camera samples from after registration, and the stale sample sits at a timestamp from before
	 * it.
	 */
	device.id.store(id, std::memory_order_release);

	this->num_devices += 1;

	os_thread_helper_unlock(&this->thread);
}

void
SensorFusion::removeDevice(t_constellation_device_id_t device_id)
{
	os_thread_helper_lock(&this->thread);

	assert(this->num_devices > 0);

	for (uint32_t fusion_device_idx = 0; fusion_device_idx < this->devices.size(); fusion_device_idx++) {
		auto &device = this->devices[fusion_device_idx];

		if (device.id.load(std::memory_order_relaxed) != device_id) {
			continue;
		}

		/*
		 * Release, so an IMU push racing this either sees the old ID and writes into a slot we are about to
		 * abandon, or sees the cleared one and skips the slot entirely. The slot is not handed out again until
		 * addDevice() has reset it.
		 */
		device.id.store(XRT_CONSTELLATION_INVALID_DEVICE_ID, std::memory_order_release);

		// Invalidate this device's prior
		this->problem->invalidatePriorLocked(fusion_device_idx);

		device.led_model = nullptr;

		this->num_devices -= 1;
		os_thread_helper_unlock(&this->thread);
		return;
	}

	os_thread_helper_unlock(&this->thread);

	assert(!"Unreachable"); // We failed to find a device slot?

	// Fallback for release mode
	throw std::runtime_error("unreachable: Failed to find device " + std::to_string(device_id) +
	                         " in device slots for removal");
}

void
SensorFusion::setDeviceImuCalibration(t_constellation_device_id_t device_id, const DeviceImuCalibration &calibration)
{
	os_thread_helper_lock(&this->thread);

	uint32_t device_fusion_idx = 0;
	if (!constellationDeviceToFusionDevice(this->devices, device_id, device_fusion_idx)) {
		os_thread_helper_unlock(&this->thread);

		SF_WARN(this, "IMU calibration for device %d, which is not registered with the fusion.", device_id);
		return;
	}

	Device &device = this->devices[device_fusion_idx];

	device.calibrated_accel_bias = calibration.accel_bias;
	device.calibrated_gyro_bias = calibration.gyro_bias;
	device.calibrated_accel_scale = calibration.accel_scale;
	device.Qcv_imu_model = calibration.Qcv_imu_model;

	/*
	 * The preintegration linearizes around these, and the calibrated values are a far better starting point than
	 * "unbiased", the first-order bias correction in the residual only has to carry the distance from here.
	 */
	device.latest_accel_bias = calibration.accel_bias;
	device.latest_gyro_bias = calibration.gyro_bias;
	device.latest_accel_scale = calibration.accel_scale;

	/*
	 * Blocks seeded from the old values, and the anchor built out of them, both belong to a calibration that no
	 * longer applies.
	 */
	this->problem->resetDeviceCalibrationSeedsLocked(device_fusion_idx);

	/*
	 * Every interval this device holds was integrated at the old linearization point, so it has to be walked
	 * again rather than corrected across a gap this large.
	 */
	for (auto &preintegration : device.preintegrations) {
		preintegration.state = DevicePreintegrationState::Dirty;
	}

	SF_INFO(this,
	        "Device %d IMU calibration set. Accel bias [%f, %f, %f] m/s^2, gyro bias [%f, %f, %f] rad/s, accel "
	        "scale [%f, %f, %f].",
	        device_id,                                                                        //
	        calibration.accel_bias.x, calibration.accel_bias.y, calibration.accel_bias.z,     //
	        calibration.gyro_bias.x, calibration.gyro_bias.y, calibration.gyro_bias.z,        //
	        calibration.accel_scale.x, calibration.accel_scale.y, calibration.accel_scale.z); //

	this->markDirtyLocked();

	os_thread_helper_unlock(&this->thread);
}

void
SensorFusion::pushCameraSample(constellation::ConstellationTracker *tracker, const constellation::CameraSample &sample)
{
	/*
	 * Only ever needed `tracker` to act as a "badge" to make sure the caller actually holds the lock on the
	 * tracker, so let's just mark it unused.
	 */
	(void)tracker;

	// Ignore samples with no camera position
	if (!sample.Txr_world_cam.has_value()) {
		return;
	}

	SF_TRACE(this, "Camera sample %u/%u at %" PRIi64 " pushed.", sample.mosaic_index, sample.camera_index,
	         sample.timestamp_ns);

	os_thread_helper_lock(&this->thread);

	if (this->sample_queue.size() >= kMaxQueuedCameraSamples) {
		this->num_queued_samples_dropped += 1;
		const uint64_t num_dropped = this->num_queued_samples_dropped;

		os_thread_helper_unlock(&this->thread);

		SF_WARN(this,
		        "Fusion thread is %u camera samples behind, dropping %u/%u at %" PRIi64 ". (%" PRIu64
		        " dropped so far)",
		        kMaxQueuedCameraSamples, sample.mosaic_index, sample.camera_index, sample.timestamp_ns,
		        num_dropped);
		return;
	}

	this->sample_queue.push_back(sample);

	this->markDirtyLocked();

	os_thread_helper_unlock(&this->thread);
}

void
SensorFusion::processCameraSampleLocked(const constellation::CameraSample &sample)
{
	Keyframe *keyframe = nullptr;
	// Only meaningful once `keyframe` is set. A freshly pushed keyframe is the newest, hence the 0.
	uint32_t keyframe_age = 0;

	/*
	 * Look for the exposure this sample belongs to. The cameras of a mosaic expose together but finish at
	 * different times, so a sample routinely arrives after a keyframe has already been opened for its exposure -
	 * and with per-camera capture threads, occasionally after a newer exposure has opened one too.
	 */
	for (uint32_t age = 0; age < this->num_keyframes; age++) {
		Keyframe &candidate = this->keyframeLocked(age);

		if (sample.mosaic_index != candidate.mosaic_idx) {
			continue;
		}

		if (std::abs(sample.timestamp_ns - candidate.timestamp_ns) >= kKeyframeGroupingToleranceNs) {
			continue;
		}

		if (candidate.sealed) {
			/*
			 * Too late to take it. Its neighbours may already have been folded into a marginalization
			 * prior, and adding a factor to it now would count this exposure's information twice.
			 */
			this->num_late_samples_dropped += 1;

			SF_DEBUG(this,
			         "Camera sample %u/%u at %" PRIi64
			         " arrived after its keyframe was sealed, dropping it. (%" PRIu64 " dropped so far)",
			         sample.mosaic_index, sample.camera_index, sample.timestamp_ns,
			         this->num_late_samples_dropped);

			return;
		}

		keyframe = &candidate;
		keyframe_age = age;
		break;
	}

	if (keyframe == nullptr) {
		/*
		 * Nothing in the window wants it and it predates the newest keyframe, so it belongs somewhere in the
		 * middle of the window, which we cannot open a keyframe in without reordering everything after it.
		 */
		if (this->num_keyframes > 0 && sample.timestamp_ns < this->keyframeLocked(0).timestamp_ns) {
			this->num_late_samples_dropped += 1;

			SF_DEBUG(this,
			         "Camera sample %u/%u at %" PRIi64
			         " is older than the newest keyframe and matches none, dropping it. (%" PRIu64
			         " dropped so far)",
			         sample.mosaic_index, sample.camera_index, sample.timestamp_ns,
			         this->num_late_samples_dropped);

			return;
		}

		keyframe = &this->pushKeyframeLocked(sample);
	}

	writeSampleToKeyframe(this, sample, *keyframe, keyframe_age);

	/*
	 * Set whether or not this camera actually saw anything: what the mask tracks is whether the exposure is
	 * complete, not whether it was useful.
	 */
	keyframe->contributed_camera_mask |= 1u << sample.camera_index;

	/*
	 * Every camera of the mosaic has reported, so nothing else can legitimately arrive for this exposure and there
	 * is no reason to hold it open until it ages out.
	 */
	if (keyframe->contributed_camera_mask == this->mosaic_camera_masks[keyframe->mosaic_idx]) {
		keyframe->sealed = true;
	}

	for (auto &device : this->devices) {
		this->updateDeviceLifecycleLocked(device);
	}
}

void
SensorFusion::pushImuSample(t_constellation_device_id_t device_id, const xrt_imu_sample &xr_sample)
{
	/*
	 * @note This deliberately runs without taking `this->thread`. We *really* do not want a heavy lock on the IMU
	 *       path, so the ring is a single-producer/single-consumer structure instead: we fill the slot first and
	 *       then publish it by bumping the count, and the fusion thread reads in the mirrored order. See
	 *       `imu_write_count` for more details.
	 */

	SF_TRACE(this, "IMU sample %" PRIi64 " for %d pushed.", xr_sample.timestamp_ns, device_id);

	// Convert from OpenXR to OpenCV coordinate space for the IMU sample
	xrt_imu_sample cv_sample = xr_sample;
	math_vec3_f64_convert_from_opencv(&cv_sample.accel_m_s2, &cv_sample.accel_m_s2);
	math_vec3_f64_convert_from_opencv(&cv_sample.gyro_rad_secs, &cv_sample.gyro_rad_secs);

	for (auto &device : this->devices) {
		/*
		 * Acquire, pairing with the release store in addDevice(): if the ID matches then the rest of the slot
		 * is set up and safe to touch. A slot whose ID does not match is never dereferenced any further, which
		 * is what keeps a device being added or removed from racing this.
		 */
		if (device.id.load(std::memory_order_acquire) != device_id) {
			continue;
		}

		// Relaxed is enough to read the count back: this thread is its only writer.
		const uint64_t count = device.imu_write_count.load(std::memory_order_relaxed);

		device.imu_samples[count % kImuBufferSize].store(cv_sample);

		/*
		 * Release, so that a reader acquiring the new count is guaranteed to see the slot we just filled and
		 * never an older or half-written one.
		 */
		device.imu_write_count.store(count + 1, std::memory_order_release);

		/*
		 * The fusion thread parked a preintegration because the IMU had not reached the end of its interval
		 * yet, and this sample may be the one that closes it. Signalled *without* the mutex on purpose: taking
		 * it here would put the IMU thread behind whatever the fusion thread is chewing on, which is the one
		 * thing this path must never do. The cost is that a wakeup can be lost to that race, which just means
		 * waiting for the next sample at most a couple milliseconds later.
		 */
		if (this->waiting_on_imu.load(std::memory_order_relaxed)) {
			os_thread_helper_signal_locked(&this->thread);
		}

		return;
	}

	// It's valid to not have the device here immediately.
	return;
}

void
SensorFusion::getTrackedPose(t_constellation_device_id_t device_id,
                             timepoint_ns requested_time_ns,
                             timepoint_ns &out_time_ns,
                             xrt_space_relation &out_relation)
{
	out_relation = XRT_SPACE_RELATION_ZERO;

	for (uint32_t fusion_device_idx = 0; fusion_device_idx < this->devices.size(); fusion_device_idx++) {
		auto &device = this->devices[fusion_device_idx];

		/*
		 * Acquire, pairing with the release store in addDevice(): if the ID matches then the rest of the slot
		 * is set up and safe to touch. A slot whose ID does not match is never dereferenced any further, which
		 * is what keeps a device being added or removed from racing this.
		 */
		if (device.id.load(std::memory_order_acquire) != device_id) {
			continue;
		}

		os_thread_helper_lock(&this->thread);

		if (!device.has_latest_solve) {
			os_thread_helper_unlock(&this->thread);
			break;
		}

		double world_gravity_mag = this->snapshot.world_gravity_mag;
		timepoint_ns latest_solve_time_ns = device.latest_solve_time_ns;
		xrt_pose Tcv_world_device = device.Tcv_world_device_latest;
		xrt_vec3_f64 world_velocity = device.latest_world_velocity;
		xrt_vec3_f64 accel_bias = device.latest_accel_bias;
		xrt_vec3_f64 gyro_bias = device.latest_gyro_bias;
		xrt_vec3_f64 accel_scale = device.latest_accel_scale;
		xrt_quat Qcv_imu_model = device.Qcv_imu_model;
		const int64_t max_dead_reckoning_ns = device.max_dead_reckoning_ns;

		os_thread_helper_unlock(&this->thread);

		/*
		 * Nothing has constrained this device for longer than it is allowed to be extrapolated for, so stop
		 * integrating where that allowance ran out and hold the pose there.
		 *
		 * The unlimited case (a limit of zero) is deliberately left exactly as it was.
		 */
		timepoint_ns integrate_until_ns = requested_time_ns;
		bool holding = false;
		if (max_dead_reckoning_ns > 0 && requested_time_ns - latest_solve_time_ns > max_dead_reckoning_ns) {
			integrate_until_ns = latest_solve_time_ns + max_dead_reckoning_ns;
			holding = true;
		}

		bool integrate_to = true;

		if (integrate_until_ns <= latest_solve_time_ns) {
			SF_TRACE(this, "Requested time is before our latest sample, can't work backwards.");
			integrate_to = false;
		}

		std::array<xrt_imu_sample, kImuBufferSize> samples_buf;
		uint32_t num_samples = 0;
		std::optional<xrt_imu_sample> first_after;

		if (integrate_to) {
			xrt_imu_sample raw_first_after;
			const ImuCollectResult result = collectImuSamplesForRange(device,               //
			                                                          latest_solve_time_ns, //
			                                                          integrate_until_ns,   //
			                                                          samples_buf,          //
			                                                          num_samples,          //
			                                                          raw_first_after);     //

			// Check if we got it OK
			if (result != ImuCollectResult::Ok && result != ImuCollectResult::NoSampleAfterEnd) {
				SF_TRACE(this, "IMU data collection for dead reckoning failed, got %d",
				         static_cast<int>(result));
				integrate_to = false;
			}

			if (num_samples == 0) {
				SF_TRACE(this, "No IMU data found for dead reckoning");
				integrate_to = false;
			}

			if (result == ImuCollectResult::NoSampleAfterEnd) {
				first_after = std::nullopt;
			} else {
				first_after = raw_first_after;
			}
		}

		if (integrate_to) {
			integrateToTimestamp(                                                 //
			    world_gravity_mag,                                                //
			    latest_solve_time_ns,                                             //
			    Tcv_world_device,                                                 //
			    world_velocity,                                                   //
			    accel_bias,                                                       //
			    gyro_bias,                                                        //
			    accel_scale,                                                      //
			    std::span<const xrt_imu_sample>(samples_buf.data(), num_samples), //
			    first_after,                                                      //
			    out_relation);                                                    //
			out_time_ns = samples_buf[num_samples - 1].timestamp_ns;

			SF_TRACE(this, "Used dead reckoning for pose at ts %" PRIi64, requested_time_ns);
		} else {
			m_relation_history_get(device.relation_history, integrate_until_ns, &out_relation);
			out_time_ns = integrate_until_ns;

			SF_TRACE(this, "Used relation history for pose at ts %" PRIi64, requested_time_ns);
		}

		math_quat_rotate(&out_relation.pose.orientation, &Qcv_imu_model, &out_relation.pose.orientation);

		// Convert to OpenXR coordinate space before outputting
		math_pose_convert_from_opencv(&out_relation.pose, &out_relation.pose);
		math_vec3_convert_from_opencv(&out_relation.linear_velocity, &out_relation.linear_velocity);
		math_vec3_convert_from_opencv(&out_relation.angular_velocity, &out_relation.angular_velocity);

		if (holding) {
			/*
			 * Held, so there is no motion to report and the caller must not predict any: `out_time_ns` is
			 * moved to the requested time so the tracker's forward prediction spans zero seconds. It is also no
			 * longer being tracked, and says so.
			 */
			out_relation.linear_velocity = XRT_VEC3_ZERO;
			out_relation.angular_velocity = XRT_VEC3_ZERO;
			out_relation.relation_flags = static_cast<xrt_space_relation_flags>(
			    out_relation.relation_flags &
			    ~(XRT_SPACE_RELATION_POSITION_TRACKED_BIT | XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT |
			      XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT | XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT));
			out_time_ns = requested_time_ns;

			SF_DEBUG(this, "Device %d is past its dead reckoning limit, holding its pose.", device_id);
		}

		SF_TRACE(this,
		         "Pose get from %" PRIi64 " to %" PRIi64
		         " succeeded.\n"
		         "Pose: %f,%f,%f\t%f,%f,%f,%f\n"
		         "Velocity: %f,%f,%f\t%f,%f,%f",
		         latest_solve_time_ns, out_time_ns, out_relation.pose.position.x, out_relation.pose.position.y,
		         out_relation.pose.position.z, out_relation.pose.orientation.x, out_relation.pose.orientation.y,
		         out_relation.pose.orientation.z, out_relation.pose.orientation.w,
		         out_relation.linear_velocity.x, out_relation.linear_velocity.y, out_relation.linear_velocity.z,
		         out_relation.angular_velocity.x, out_relation.angular_velocity.y,
		         out_relation.angular_velocity.z);

		return;
	}
}

void
SensorFusion::reset()
{
	os_thread_helper_lock(&this->thread);

	/*
	 * Anything already queued describes the window we are throwing away, so it goes with it rather than being
	 * folded into the fresh one. So does the prior, which is a statement about keyframes that are going away.
	 */
	this->sample_queue.clear();
	for (uint32_t fusion_device_idx = 0; fusion_device_idx < this->devices.size(); fusion_device_idx++) {
		this->problem->invalidatePriorLocked(fusion_device_idx);
	}

	/*
	 * Dropping the window is a counter reset. Nothing reads a slot beyond `num_keyframes`, so the stale
	 * contents are unreachable and do not need wiping.
	 */
	this->keyframe_head = 0;
	this->num_keyframes = 0;

	this->dirty = false;
	this->waiting_on_imu.store(false, std::memory_order_relaxed);
	this->num_late_samples_dropped = 0;

	for (auto &device : this->devices) {
		if (device.id.load(std::memory_order_relaxed) == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
			continue;
		}

		/*
		 * Drops the buffered IMU history. Release so that the IMU thread cannot observe the reset count
		 * alongside stale slot contents. This does race with a push already in flight, which would restore the
		 * old count. This is harmless, since samples are filtered by timestamp and the stale ones fall outside
		 * any keyframe range we go on to build.
		 */
		device.imu_write_count.store(0, std::memory_order_release);

		device.tracking_state = DeviceTrackingState::Uninitialized;
		device.num_observed_keyframes = 0;

		for (auto &preintegration : device.preintegrations) {
			preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Dirty);
		}
	}

	os_thread_helper_unlock(&this->thread);
}

bool
SensorFusion::tickLocked(bool have_new_input)
{
	if (!this->dirty) {
		return false;
	}

	/*
	 * Set by anything that wants to be run again, so that a preintegration left unfinished is retried rather than
	 * silently forgotten until the next camera sample happens along.
	 */
	bool work_remaining = false;
	/*
	 * Narrower than the above: only work that a *new IMU sample* can unblock, which is the only work worth having
	 * the IMU path wake us up for.
	 */
	bool wants_imu_wakeup = false;

	for (uint32_t device_fusion_idx = 0; device_fusion_idx < this->devices.size(); device_fusion_idx++) {
		auto &device = this->devices[device_fusion_idx];

		const t_constellation_device_id_t device_id = device.id.load(std::memory_order_relaxed);

		if (device_id == XRT_CONSTELLATION_INVALID_DEVICE_ID) {
			continue;
		}

		if (device.tracking_state == DeviceTrackingState::Lost) {
			SF_DEBUG(this, "Dropping window state for lost device %d.", device_id);

			// This device is lost, so the prior we had is no longer valid.
			this->problem->invalidatePriorLocked(device_fusion_idx);

			// @todo Once the optimizer keeps per-keyframe device states, this is where they have to be
			//       thrown away too.
			device.tracking_state = DeviceTrackingState::Uninitialized;
		}

		/*
		 * Nothing in the window observes this device, so its IMU chain has no states to connect and integrating
		 * it now would be work thrown away.
		 */
		if (device.num_observed_keyframes == 0) {
			continue;
		}

		ImuBias<ImuBiasJet> bias = ImuBias<double>(map_vec3_f64(device.latest_accel_bias),  //
		                                           map_vec3_f64(device.latest_gyro_bias),   //
		                                           map_vec3_f64(device.latest_accel_scale)) //
		                               .seed();

		/*
		 * `age` names the keyframe an interval *ends* at, so the oldest keyframe in the window is skipped:
		 * there is no predecessor left in the window to integrate it from.
		 */
		for (uint32_t end_age = 0; end_age + 1 < this->num_keyframes; end_age++) {
			auto &preintegration = this->preintegrationLocked(device, end_age);

			// Doesn't need to be worked on
			if (!preintegration.dirty()) {
				continue;
			}

			uint32_t start_age = end_age + 1;

			const auto &end_keyframe = this->keyframeLocked(end_age);

			if (!(end_keyframe.observed_device_mask & (1u << device_fusion_idx))) {
				preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Dirty);
				continue;
			}

			Keyframe *start_keyframe = &this->keyframeLocked(start_age);
			while (!(start_keyframe->observed_device_mask & (1u << device_fusion_idx)) &&
			       (start_age + 1 < this->num_keyframes)) {
				start_keyframe = &this->keyframeLocked(start_age + 1);
				start_age++;
			}

			if (!(start_keyframe->observed_device_mask & (1u << device_fusion_idx))) {
				// There is no earlier keyframe for this to attach to.
				preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Empty);
				continue;
			}

			if (end_keyframe.timestamp_ns == start_keyframe->timestamp_ns) {
				/*
				 * Nothing to integrate over. Identity carries a zero whitening matrix, so the factor
				 * contributes nothing rather than contributing something wrong.
				 */
				preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Empty);
				continue;
			}

			assert(end_keyframe.timestamp_ns > start_keyframe->timestamp_ns);

			std::array<xrt_imu_sample, kMaxImuSamplesPerKeyframe> samples_buf;
			uint32_t num_samples = 0;
			xrt_imu_sample first_after;

			const ImuCollectResult result = collectImuSamplesForRange(device,                       //
			                                                          start_keyframe->timestamp_ns, //
			                                                          end_keyframe.timestamp_ns,    //
			                                                          samples_buf,                  //
			                                                          num_samples,                  //
			                                                          first_after);                 //

			switch (result) {
			case ImuCollectResult::Ok: break;

			case ImuCollectResult::Lapped:
				SF_WARN(this, "IMU ring lapped while collecting samples, fusion is running slow.");
				/*
				 * A retry can succeed once the fusion thread has caught up, but hammering it from the
				 * IMU path would only make it slower, so this waits for the next camera sample.
				 */
				work_remaining = true;
				continue;

			case ImuCollectResult::Overflow:
				SF_WARN(this,
				        "More than %u IMU samples between keyframes at %" PRIi64 " and %" PRIi64
				        ", leaving the interval unconstrained.",
				        kMaxImuSamplesPerKeyframe, start_keyframe->timestamp_ns,
				        end_keyframe.timestamp_ns);

				/*
				 * Retrying cannot help, since the range only ever grows. Make the interval explicitly
				 * empty so that it's not going to be checked again.
				 */
				preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Empty);
				continue;

			case ImuCollectResult::NoSampleAfterEnd:
				/*
				 * The IMU has not reached the end of this interval yet. The sample that lands past it
				 * is what unblocks this, so ask pushImuSample() to wake us when one does.
				 */
				work_remaining = true;
				wants_imu_wakeup = true;
				continue;
			}

			preintegration.relative_start_age = start_age - end_age;

			/*
			 * preintegrate() needs at least a single sample to run correctly.
			 *
			 * @todo Have it work with zero samples if we have an IMU sample *after*.
			 */
			if (num_samples < 1) {
				SF_DEBUG(this,
				         "Only %u IMU samples between keyframes at %" PRIi64 " and %" PRIi64
				         ", leaving the interval unconstrained.",
				         num_samples, start_keyframe->timestamp_ns, end_keyframe.timestamp_ns);

				preintegration = DevicePreintegration::Identity(DevicePreintegrationState::Empty);
				continue;
			}

			preintegration.imu_preintegration =
			    preintegrate(std::span<const xrt_imu_sample>(samples_buf.data(),
			                                                 num_samples), //
			                 first_after,                                  //
			                 bias,                                         //
			                 start_keyframe->timestamp_ns,                 //
			                 end_keyframe.timestamp_ns);                   //

			/*
			 * A keyframe this integrated straight over can still gain an observation of the device and
			 * force the interval to be cut in two, but only while it is open. Once every skipped keyframe
			 * is sealed nothing can land in the gap any more, so the interval is final. With no gaps at
			 * all the inner loop does not run and this seals immediately.
			 */
			bool any_skipped_keyframe_open = false;
			for (uint32_t skipped_age = end_age + 1; skipped_age < start_age; skipped_age++) {
				if (!this->keyframeLocked(skipped_age).sealed) {
					any_skipped_keyframe_open = true;
					break;
				}
			}

			preintegration.state = any_skipped_keyframe_open ? DevicePreintegrationState::MayNeedSplit
			                                                 : DevicePreintegrationState::Sealed;

			SF_TRACE(this, "Preintegrated IMU between %" PRIi64 " and %" PRIi64 ". (dt %lf)",
			         start_keyframe->timestamp_ns, end_keyframe.timestamp_ns,
			         preintegration.imu_preintegration.dt);
		}
	}

	this->waiting_on_imu.store(wants_imu_wakeup, std::memory_order_relaxed);

	/*
	 * Deliberately not gated on `work_remaining`. buildLocked() already leaves an interval that has not integrated
	 * out of the problem, so the worst an unfinished one costs is that its device's newest keyframe sits out this
	 * solve and joins the next.
	 */
	bool run_fusion = false;
	if (have_new_input) {
		run_fusion = this->buildFusionLocked();
	}

	this->dirty = work_remaining;

	return run_fusion;
}

}; // namespace xrt::tracking::constellation::optimizer::sensor_fusion
