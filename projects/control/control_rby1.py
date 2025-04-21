import rby1_sdk
import numpy as np
import sys
import time
import argparse
import re
from rby1_sdk import *

D2R = np.pi / 180  # Degree to Radian conversion factor
MINIMUM_TIME = 2.5
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-5
STOP_POSITION_TRACKING_ERROR = 1e-5
WEIGHT = 0.0015
STOP_COST = 1e-2
VELOCITY_TRACKING_GAIN = 0.01
MIN_DELTA_COST = 1e-4
PATIENCE = 10

def cb(rs):
    print(f"Timestamp: {rs.timestamp - rs.ft_sensor_right.time_since_last_update}")
    position = rs.position * 180 / 3.141592
    print(f"waist [deg]: {position[2:2 + 6]}")
    print(f"right arm [deg]: {position[8:8 + 7]}")
    print(f"left arm [deg]: {position[15:15 + 8]}")

def init_joint_position_command(robot):
    print("init joint position command")

    # Initialize joint positions
    q_joint_waist = np.zeros(6)
    q_joint_right_arm = np.zeros(7)
    q_joint_left_arm = np.zeros(7)

    # Set specific joint positions
    # q_joint_right_arm[0] = -90 * D2R
    # q_joint_left_arm[1] = 90 * D2R
    q_joint_right_arm = np.array([-45, -45, 30, -45, 20, -20, 0]) * D2R
    q_joint_left_arm = np.array([-45, 45, -30, -45, -20, -20, 0]) * D2R

    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_waist)
            )
            .set_right_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_right_arm)
            )
            .set_left_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_left_arm)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0

def mobility_command(robot):
    print("mobility command")

    q_joint_left_wheel = -1.5
    q_joint_right_wheel = -1.5

    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder()
        .set_mobility_command(
            MobilityCommandBuilder()
            .set_command(
                JointVelocityCommandBuilder()
                .set_velocity([q_joint_left_wheel, q_joint_right_wheel])
                .set_command_header(
                    CommandHeaderBuilder().set_control_hold_time(7.0)
                )
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0

def main(address, power_device, servo):
    print("Attempting to connect to the robot...")

    robot = rby1_sdk.create_robot_a(address)

    if not robot.connect():
        print("Error: Unable to establish connection to the robot at")
        sys.exit(1)

    print("Successfully connected to the robot")

    print("Starting state update...")
    robot.start_state_update(cb, 0.1)

    robot.set_parameter("default.acceleration_limit_scaling", "0.8")
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("cartesian_command.cutoff_frequency", "5")
    robot.set_parameter("default.linear_acceleration_limit", "5")
    # robot.set_time_scale(1.0)

    print("parameters setting is done")

    if not robot.is_connected():
        print("Robot is not connected")
        exit(1)
        
    if not robot.is_power_on(power_device):
        rv = robot.power_on(power_device)
        if not rv:
            print("Failed to power on")
            exit(1)
            
    print(servo)
    if not robot.is_servo_on(servo):
        rv = robot.servo_on(servo)
        if not rv:
            print("Fail to servo on")
            exit(1)

    
    control_manager_state = robot.get_control_manager_state()

    if (control_manager_state.state == rby1_sdk.ControlManagerState.State.MinorFault or control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault):

        if control_manager_state.state == rby1_sdk.ControlManagerState.State.MajorFault:
            print("Warning: Detected a Major Fault in the Control Manager!!!!!!!!!!!!!!!.")
        else:
            print("Warning: Detected a Minor Fault in the Control Manager@@@@@@@@@@@@@@@@.")

        print("Attempting to reset the fault...")
        if not robot.reset_fault_control_manager():
            print("Error: Unable to reset the fault in the Control Manager.")
            sys.exit(1)
        print("Fault reset successfully.")

    print("Control Manager state is normal. No faults detected.")

    print("Enabling the Control Manager...")
    if not robot.enable_control_manager():
        print("Error: Failed to enable the Control Manager.")
        sys.exit(1)
    print("Control Manager enabled successfully.")

    if not init_joint_position_command(robot):
        print("finish motion")
    if not mobility_command(robot):
        print("finish motion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="07_impedance_control")
    parser.add_argument('--address', type=str, default="localhost:50051", required=False, help="Robot address")
    parser.add_argument('--device', type=str, default=".*", help="Power device name regex pattern (default: '.*')")
    parser.add_argument('--servo', type=str, default=".*", help="Servo name regex pattern (default: '.*')")
    args = parser.parse_args()

    main(address=args.address,
         power_device=args.device,
         servo = args.servo)



