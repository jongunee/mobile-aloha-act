import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

MODEL_XML_PATH = "../models/rainbow_robotics_rby1/rby1.xml"

def load_and_view():
    """
    Mujoco Viewer에서 rby1.xml을 로드하여 로봇과 테이블의 위치 확인
    """
    # Mujoco 모델 로드
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    # Mujoco Viewer 실행 (환경을 렌더링)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Mujoco Viewer에서 로봇 위치를 확인하세요.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    load_and_view()