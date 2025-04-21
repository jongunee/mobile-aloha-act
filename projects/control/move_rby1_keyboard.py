import rby1_sdk
import numpy as np
import sys
import time
import argparse
import termios
import tty
import sys
import select
from rby1_sdk import *

D2R = np.pi / 180  # Degree to Radian 변환 상수
WHEEL_SPEED = 2.0  # 기본 휠 속도 (좀 더 빠르게 설정)
TURN_SPEED = 2.0   # 회전 속도
STOP_SPEED = 0.0   # 정지 속도
UPDATE_RATE = 0.02  # 입력 처리 속도 (0.1 → 0.02초로 빠르게 조정)

def get_key():
    """ 키보드 입력을 받는 함수 (sudo 없이 사용 가능) """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], UPDATE_RATE)
        if rlist:
            key = sys.stdin.read(1)  # 알파벳 키는 1바이트 입력
        else:
            key = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def move_robot(robot, left_speed, right_speed, duration=0.1):
    """로봇의 바퀴를 조정하는 함수"""
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_mobility_command(
            MobilityCommandBuilder().set_command(
                JointVelocityCommandBuilder()
                .set_velocity([left_speed, right_speed])
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(duration))
            )
        )
    )
    robot.send_command(rc, 10).get()
    time.sleep(duration)

def control_robot_with_wsad(robot):
    """ WSAD 키 입력을 통해 로봇 이동 제어 """
    print("▶ WSAD 키로 로봇 이동 조작")
    print("🚀 W: 전진 | S: 후진 | A: 좌회전 | D: 우회전 | Q: 종료")

    while True:
        key = get_key()

        if key == 'w':  # 전진
            print("🚗 전진")
            move_robot(robot, -WHEEL_SPEED, -WHEEL_SPEED, UPDATE_RATE)
        elif key == 's':  # 후진
            print("⏪ 후진")
            move_robot(robot, WHEEL_SPEED, WHEEL_SPEED, UPDATE_RATE)
        elif key == 'a':  # 좌회전
            print("↩️ 좌회전")
            move_robot(robot, -TURN_SPEED, TURN_SPEED, UPDATE_RATE)
        elif key == 'd':  # 우회전
            print("↪️ 우회전")
            move_robot(robot, TURN_SPEED, -TURN_SPEED, UPDATE_RATE)
        elif key == 'q':  # 종료
            print("🛑 이동 종료")
            move_robot(robot, STOP_SPEED, STOP_SPEED)
            break

def main(address, power_device, servo):
    print("Attempting to connect to the robot...")

    robot = rby1_sdk.create_robot_a(address)

    if not robot.connect():
        print("Error: Unable to establish connection to the robot")
        sys.exit(1)

    print("Successfully connected to the robot")

    print("Starting state update...")
    robot.start_state_update(lambda rs: None, UPDATE_RATE)  # 상태 업데이트 속도 향상

    # 전원 및 서보 활성화
    if not robot.is_power_on(power_device):
        print("Power is OFF! Turning it ON...")
        robot.power_on(power_device)

    if not robot.is_servo_on(servo):
        print("Servo is OFF! Turning it ON...")
        robot.servo_on(servo)

    print("Control Manager state is normal. No faults detected.")

    print("Enabling the Control Manager...")
    if not robot.enable_control_manager():
        print("Error: Failed to enable the Control Manager.")
        sys.exit(1)
    print("Control Manager enabled successfully.")

    # WSAD 키 입력을 통한 이동 시작
    control_robot_with_wsad(robot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBY1 WSAD Keyboard Control")
    parser.add_argument('--address', type=str, default="localhost:50051", required=False, help="Robot address")
    parser.add_argument('--device', type=str, default=".*", help="Power device name regex pattern")
    parser.add_argument('--servo', type=str, default=".*", help="Servo name regex pattern")
    args = parser.parse_args()

    main(address=args.address,
         power_device=args.device,
         servo=args.servo)
