import mujoco
import mujoco.viewer
import numpy as np
import time
import termios
import tty
import sys
import select

# Mujoco 모델 경로
MODEL_XML_PATH = "../models/rainbow_robotics_rby1/rby1.xml"

# 이동 속도 설정
WHEEL_SPEED = 1.0  # 바퀴 속도
TURN_SPEED = 1.0   # 회전 속도
STOP_SPEED = 0.0   # 정지 속도
UPDATE_RATE = 0.02  # 입력 처리 속도 (빠른 반응 속도)

# RBY1 로봇의 바퀴 액추에이터 인덱스 (출력된 인덱스 기반으로 수정!)
RIGHT_WHEEL_CTRL = 2  # 오른쪽 바퀴 액추에이터
LEFT_WHEEL_CTRL = 3   # 왼쪽 바퀴 액추에이터

def get_key():
    """ 키보드 입력을 받는 함수 (sudo 없이 사용 가능) """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], UPDATE_RATE)
        if rlist:
            key = sys.stdin.read(1)  # 단일 문자 입력 받기
        else:
            key = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def move_robot(data, left_speed, right_speed):
    """ Mujoco 환경에서 로봇 바퀴 속도 적용 """
    data.ctrl[LEFT_WHEEL_CTRL] = left_speed  # 왼쪽 바퀴 속도
    data.ctrl[RIGHT_WHEEL_CTRL] = right_speed  # 오른쪽 바퀴 속도

def control_robot_with_wsad(model, data):
    """ WSAD 키 입력을 통해 Mujoco 환경의 로봇 이동 제어 """
    print("▶ WSAD 키로 Mujoco 환경의 RBY1 로봇 이동 조작")
    print("🚀 W: 전진 | S: 후진 | A: 좌회전 | D: 우회전 | Q: 종료")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            key = get_key()

            if key == 'w':  # 전진
                print("🚗 전진")
                move_robot(data, -WHEEL_SPEED, -WHEEL_SPEED)
            elif key == 's':  # 후진
                print("⏪ 후진")
                move_robot(data, WHEEL_SPEED, WHEEL_SPEED)
            elif key == 'a':  # 좌회전
                print("↩️ 좌회전")
                move_robot(data, -TURN_SPEED, TURN_SPEED)
            elif key == 'd':  # 우회전
                print("↪️ 우회전")
                move_robot(data, TURN_SPEED, -TURN_SPEED)
            elif key == 'q':  # 종료
                print("🛑 이동 종료")
                move_robot(data, STOP_SPEED, STOP_SPEED)
                break

            mujoco.mj_step(model, data)  # Mujoco 물리 시뮬레이션 업데이트
            viewer.sync()
            time.sleep(UPDATE_RATE)

def main():
    """ Mujoco 환경에서 RBY1 로봇을 로드하고 제어 """
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    print("Mujoco 환경을 실행합니다...")
    control_robot_with_wsad(model, data)

if __name__ == "__main__":
    main()
