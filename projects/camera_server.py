import mujoco
import numpy as np
import cv2
import os
import time
from flask import Flask, send_file
from threading import Thread
from constants_rby1 import XML_DIR

app = Flask(__name__)
image_path = "/tmp/top.jpg"
D2R = np.pi / 180

def load_model_and_data(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

def initialize_pose(model, data):
    D2R = np.pi / 180
    joint_names = [
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3',
        'right_arm_4', 'right_arm_5', 'right_arm_6',
        'left_arm_0',  'left_arm_1',  'left_arm_2',  'left_arm_3',
        'left_arm_4',  'left_arm_5',  'left_arm_6'
    ]
    target_degrees = [-45, -45, 30, -45, 20, -20, 0,
                      -45, 45, -30, -45, -20, -20, 0]
    for name, deg in zip(joint_names, target_degrees):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = deg * D2R

    # mocap 설정 (ID 3, 4는 관례적 값이며 모델에 따라 다를 수 있음)
    data.mocap_pos[3] = np.array([0.3, 0.2, 1.2])
    data.mocap_quat[3] = np.array([1, 0, 0, 0])
    data.mocap_pos[4] = np.array([-0.3, 0.2, 1.2])
    data.mocap_quat[4] = np.array([1, 0, 0, 0])



def render_loop(model, data, camera_name="top", width=640, height=480):
    renderer = mujoco.Renderer(model, height=height, width=width)
    while True:
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_name)
        rgb_img = renderer.render()
        rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, rgb_img_bgr)
        time.sleep(0.05)  # 20fps

@app.route("/top.jpg")
def send_camera_image():
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == "__main__":
    xml_path = os.path.join(XML_DIR, "rby1.xml")
    model, data = load_model_and_data(xml_path)
    initialize_pose(model, data)

    # 렌더링 백그라운드 실행
    render_thread = Thread(target=render_loop, args=(model, data), daemon=True)
    render_thread.start()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=8999)
