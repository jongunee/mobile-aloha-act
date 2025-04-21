import mujoco
import json
import numpy as np

# Mujoco 모델 경로 (RBY1 환경 XML 파일)
MODEL_XML_PATH = "../models/rainbow_robotics_rby1/rby1.xml"
OUTPUT_JSON = "rby1_joint_info.json"

def get_joint_info():
    """ Mujoco 모델의 모든 조인트 정보를 JSON 형태로 저장 """
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    
    joint_info = []
    for i in range(model.njnt):
        joint_data = {
            "index": int(i),  # NumPy int → Python int 변환
            "name": model.jnt(i).name,  # 조인트 이름
            "type": int(model.jnt_type[i]),  # NumPy int → Python int 변환
            "range": list(map(float, model.jnt_range[i])) if model.jnt_limited[i] else None  # NumPy float → Python float 변환
        }
        joint_info.append(joint_data)

    # JSON 파일로 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(joint_info, f, indent=4, ensure_ascii=False)

    print(f"✅ 조인트 정보가 '{OUTPUT_JSON}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    get_joint_info()
