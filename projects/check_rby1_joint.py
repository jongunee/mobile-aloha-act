import os
import numpy as np
from dm_control import mujoco
from constants_rby1 import XML_DIR  # ✅ XML 파일 경로

# ✅ MuJoCo XML 불러오기
xml_path = os.path.join(XML_DIR, "rby1.xml")
physics = mujoco.Physics.from_xml_path(xml_path)

# ✅ 모든 조인트의 이름과 인덱스 출력
print("\n=== RBY1 조인트 인덱스 목록 ===")
for i in range(physics.model.nq):  # ✅ 전체 qpos 크기만큼 반복
    joint_name = physics.model.id2name(i, 'actuator')
    if joint_name is None:
        joint_name = f"Unnamed_Joint_{i}"  # ✅ 이름이 없으면 임시 이름 부여
    print(f"Index {i}: {joint_name}")

# ✅ 총 조인트 개수 확인
print(f"\nTotal joints recognized: {physics.model.njnt}")
