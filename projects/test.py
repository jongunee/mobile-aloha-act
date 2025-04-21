import os
import matplotlib.pyplot as plt
from dm_control import mujoco
from constants_rby1 import XML_DIR


# XML 파일이 있는 디렉토리 경로 (필요에 따라 수정)
xml_path = os.path.join(XML_DIR, "rby1.xml")

# MuJoCo 모델 불러오기
physics = mujoco.Physics.from_xml_path(xml_path)

# 시뮬레이션 실행: Physics 객체의 step() 메서드를 사용합니다.
for _ in range(100):
    physics.step()  # 기본 1 스텝 진행

# 상단 카메라 "top"으로 렌더링
img = physics.render(height=480, width=640, camera_id='top')

plt.imshow(img)
plt.title("Top View: Ground Alignment Check")
plt.axis('off')
plt.show()
