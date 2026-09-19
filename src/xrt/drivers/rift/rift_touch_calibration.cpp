// Copyright 2025, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Code to parse and handle the Rift Touch configuration data.
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup drv_rift
 */

#include "util/u_json.hpp"
#include "util/u_logging.h"
#include "util/u_debug.h"

#include "math/m_api.h"
#include "math/m_vec3.h"

#include "rift_internal.h"

#include <string.h>
#include <math.h>


using xrt::auxiliary::util::json::JSONNode;


extern "C" bool
rift_touch_calibration_parse(const char *calibration_data,
                             size_t calibration_size,
                             struct rift_touch_controller_calibration *out_calibration)
{
	memset(out_calibration, 0, sizeof(*out_calibration));

	try {
		JSONNode root{std::string{calibration_data, calibration_size}};
		if (root.isInvalid()) {
			U_LOG_E("Failed to parse JSON config data.");
			return false;
		}

		auto tracked_object = root["TrackedObject"];

		if (tracked_object["JsonVersion"].asInt() != 2) {
			U_LOG_E("Unsupported JSON version %d", tracked_object["JsonVersion"].asInt());
			return false;
		}

		out_calibration->joy_x_range[0] = static_cast<uint16_t>(tracked_object["JoyXRangeMin"].asInt());
		out_calibration->joy_x_range[1] = static_cast<uint16_t>(tracked_object["JoyXRangeMax"].asInt());
		out_calibration->joy_x_dead[0] = static_cast<uint16_t>(tracked_object["JoyXDeadMin"].asInt());
		out_calibration->joy_x_dead[1] = static_cast<uint16_t>(tracked_object["JoyXDeadMax"].asInt());

		out_calibration->joy_y_range[0] = static_cast<uint16_t>(tracked_object["JoyYRangeMin"].asInt());
		out_calibration->joy_y_range[1] = static_cast<uint16_t>(tracked_object["JoyYRangeMax"].asInt());
		out_calibration->joy_y_dead[0] = static_cast<uint16_t>(tracked_object["JoyYDeadMin"].asInt());
		out_calibration->joy_y_dead[1] = static_cast<uint16_t>(tracked_object["JoyYDeadMax"].asInt());

		out_calibration->trigger_range[0] = static_cast<uint16_t>(tracked_object["TriggerMinRange"].asInt());
		out_calibration->trigger_range[1] = static_cast<uint16_t>(tracked_object["TriggerMidRange"].asInt());
		out_calibration->trigger_range[2] = static_cast<uint16_t>(tracked_object["TriggerMaxRange"].asInt());

		out_calibration->middle_range[0] = static_cast<uint16_t>(tracked_object["MiddleMinRange"].asInt());
		out_calibration->middle_range[1] = static_cast<uint16_t>(tracked_object["MiddleMidRange"].asInt());
		out_calibration->middle_range[2] = static_cast<uint16_t>(tracked_object["MiddleMaxRange"].asInt());
		out_calibration->middle_flipped = tracked_object["MiddleFlipped"].asBool();

		auto cap_sense_min = tracked_object["CapSenseMin"];
		auto cap_sense_touch = tracked_object["CapSenseTouch"];
		for (int i = 0; i < 8; i++) {
			out_calibration->cap_sense_min[i] = static_cast<uint16_t>(cap_sense_min[i].asInt());
			out_calibration->cap_sense_touch[i] = static_cast<uint16_t>(cap_sense_touch[i].asInt());
		}

		auto gyro_calibration_array = tracked_object["GyroCalibration"];
		for (int i = 0; i < 9; i++) {
			out_calibration->gyro_calibration[i / 3][i % 3] = gyro_calibration_array[i].asDouble();
		}
		out_calibration->gyro_offset.x = gyro_calibration_array[9 + 0].asDouble();
		out_calibration->gyro_offset.y = gyro_calibration_array[9 + 1].asDouble();
		out_calibration->gyro_offset.z = gyro_calibration_array[9 + 2].asDouble();

		auto accel_calibration_array = tracked_object["AccCalibration"];
		for (int i = 0; i < 9; i++) {
			out_calibration->accel_calibration[i / 3][i % 3] = accel_calibration_array[i].asDouble();
		}
		out_calibration->accel_offset.x = accel_calibration_array[9 + 0].asDouble();
		out_calibration->accel_offset.y = accel_calibration_array[9 + 1].asDouble();
		out_calibration->accel_offset.z = accel_calibration_array[9 + 2].asDouble();

		auto imu_position_array = tracked_object["ImuPosition"];
		out_calibration->imu_position.x = imu_position_array[0].asDouble();
		out_calibration->imu_position.y = imu_position_array[1].asDouble();
		out_calibration->imu_position.z = imu_position_array[2].asDouble();

		auto leds = tracked_object["ModelPoints"];
		out_calibration->led_model.led_count = leds.asObject().size();
		out_calibration->led_model.leds =
		    U_TYPED_ARRAY_CALLOC(struct t_constellation_tracker_led, out_calibration->led_model.led_count);

		for (size_t i = 0; i < out_calibration->led_model.led_count; i++) {
			auto led_object = leds["Point" + std::to_string(i)];

			auto &led = out_calibration->led_model.leds[i];

			led.position.x = (float)led_object[0].asDouble();
			led.position.y = (float)led_object[1].asDouble();
			led.position.z = (float)led_object[2].asDouble();
			led.normal.x = (float)led_object[3].asDouble();
			led.normal.y = (float)led_object[4].asDouble();
			led.normal.z = (float)led_object[5].asDouble();
			// The constellation tracker only takes one visibility angle, and precisely what this vector
			// means is unclear, so for now we hardcode the visibility angle below.
			/*
			led.visibility_angle.x = (float)led_object[6].asDouble();
			led.visibility_angle.y = (float)led_object[7].asDouble();
			led.visibility_angle.z = (float)led_object[8].asDouble();
			*/

			led.visibility_angle = RIFT_LED_VISIBILITY_RAD;
			led.radius_m = RIFT_LED_SIZE_M;
			led.id = i;
		}

		/*
		 * The constellation tracker's sensor fusion tracks the IMU, and only accounts for a rotation between the
		 * IMU and the LED model, never a translation. So the LED model handed to it has to be centred on the IMU,
		 * with the same axes as the IMU.
		 *
		 * The rest of the driver, though, thinks in terms of a "base" frame: the grip and aim poses are defined
		 * relative to it. Rather than bake that into the LEDs, keep it as the fixed transform @ref
		 * rift_touch_controller_calibration::T_imu_base and apply it to the tracked pose afterwards.
		 */
		struct xrt_vec3 imu_position = out_calibration->imu_position;

		// A controller is a few centimeters across. Anything larger means this isn't in the same units and frame
		// as the model points, and applying it would fling the LEDs across the room, so it is better ignored.
		if (m_vec3_len(imu_position) > 0.5f) {
			U_LOG_W("Touch controller IMU position (%f, %f, %f) is implausible, assuming it is at the origin.",
			        (double)imu_position.x, (double)imu_position.y, (double)imu_position.z);
			imu_position = {0.0f, 0.0f, 0.0f};
			out_calibration->imu_position = imu_position;
		}

		for (size_t i = 0; i < out_calibration->led_model.led_count; i++) {
			out_calibration->led_model.leds[i].position =
			    m_vec3_sub(out_calibration->led_model.leds[i].position, imu_position);
		}

		/*
		 * Where the base frame sits, relative to the model frame the calibration file gave us. Its origin is
		 * offset from the model origin, and it is tilted, in order to make our grip pose and aim pose values
		 * work nicely.
		 *
		 * A point p in the calibration file's frame is at R * (p + base_offset) in the base frame, and the model
		 * frame is that file frame shifted so the IMU is at its origin.
		 */
		struct xrt_vec3 unit_x = XRT_VEC3_UNIT_X;
		struct xrt_vec3 base_offset = {0.0f, -0.049424f, 0.02826f};

		struct xrt_pose T_base_imu = XRT_POSE_IDENTITY;
		math_quat_from_angle_vector(DEG_TO_RAD(-45.0f), &unit_x, &T_base_imu.orientation);

		struct xrt_vec3 offset_from_imu = m_vec3_add(base_offset, imu_position);
		math_quat_rotate_vec3(&T_base_imu.orientation, &offset_from_imu, &T_base_imu.position);

		math_pose_invert(&T_base_imu, &out_calibration->T_imu_base);

	} catch (const std::exception &e) {
		U_LOG_E("Exception while parsing touch controller calibration JSON: %s", e.what());
		return false;
	}

	return true;
}
