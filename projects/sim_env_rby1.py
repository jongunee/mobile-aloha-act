import numpy as np
import collections
import os
from constants_rby1 import DT, XML_DIR
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import IPython

e = IPython.embed

D2R = np.pi / 180
BOX_POSE = [None]  # 외부에서 박스 위치 설정 가능

def make_sim_env(task_name):
    """
    환경 생성: RBY1 로봇 시뮬레이션 (조인트 제어 방식)
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'rby1.xml')  # ✅ RBY1 환경 사용
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)  # ✅ 새로운 Task 클래스
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

def is_contacted(physics, geom1, geom2):
    """각 geom이 붙어 있는지 확인하는 함수"""
    all_contact_pairs = []
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_geom_1 = physics.model.id2name(id_geom_1, "geom")
        name_geom_2 = physics.model.id2name(id_geom_2, "geom")
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    # print("현재 충돌 중인 Geom 쌍:", all_contact_pairs)  
    # 박스와 테이블 접촉 여부 반환
    return (geom1, geom2) in all_contact_pairs or \
           (geom2, geom1) in all_contact_pairs
# Box 위치 무작위 샘플링 함수
def random_box_pose():
    # x, y는 책상 위 범위 내에서 무작위로 설정
    x = np.random.uniform(0.15, 0.25)
    y = np.random.uniform(0.25, 0.35)
    z = 0.94  # 고정
    quat = np.array([1, 0, 0, 0])  # 단위 쿼터니언
    return np.array([x, y, z, *quat])

def set_box_pose(env, box_pose):
    qpos = env.physics.named.data.qpos
    qpos['red_box_joint'][:7] = box_pose  # box의 이름이 'box'라고 가정

class RBY1Task(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def get_qpos(self, physics):
        """로봇의 조인트 위치를 가져옴"""
        return physics.data.qpos.copy()  # ✅ 모든 qpos 반환

    def get_qvel(self, physics):
        """로봇의 조인트 속도를 가져옴"""
        return physics.data.qvel.copy()  # ✅ 모든 qvel 반환

    def before_step(self, action, physics):
        self._step_count += 1
        """
        action: 16차원 (왼손 xyz+quat 7, 오른손 xyz+quat 7, 그리퍼 2개)
        - 0~2:   오른손 xyz
        - 3~6:   오른손 quat
        - 7~9:   왼손 xyz
        - 10~13: 왼손 quat
        - 14:    오른그리퍼
        - 15:    왼그리퍼
        """
        right_xyz = action[0:3]
        right_quat = action[3:7]
        left_xyz = action[7:10]
        left_quat = action[10:14]
        right_gripper = action[14]
        left_gripper = action[15]

        # ✅ (1) Mocap을 통해 엔드이펙터 위치 제어
        np.copyto(physics.data.mocap_pos[3], right_xyz)
        np.copyto(physics.data.mocap_quat[3], right_quat)
        np.copyto(physics.data.mocap_pos[4], left_xyz)
        np.copyto(physics.data.mocap_quat[4], left_quat)

        # ✅ (2) 26차원 ctrl 배열 준비: 0으로 초기화 후 그리퍼 2개만 값을 넣기 - 액츄에이터 번호
        ctrl_26 = np.zeros(26)  
        ctrl_26[24] = right_gripper
        ctrl_26[25] = left_gripper

        # ✅ (3) 이제 ctrl_26을 physics.data.ctrl에 복사
        np.copyto(physics.data.ctrl, ctrl_26)

        # ✅ (4) 박스-테이블 접촉 여부 확인
        # box_touching_table = is_contacted(physics, "red_box", "tabletop")

        # ✅ 디버깅 메시지 출력
        # print(f"[DEBUG] Step: {self._step_count}")
        # print(f"[DEBUG] 박스-테이블 접촉 상태: {'✅ 붙어 있음' if box_touching_table else '❌ 떨어짐'}")

        # ✅ 박스가 테이블에서 떨어졌으면 경고 메시지
        # if not box_touching_table:
        #     print("[⚠ 경고] 박스가 테이블에서 떨어졌음!")

        # 🔴 디버깅: 값이 제대로 들어가는지 확인
        # print(f"[DEBUG] Right Gripper Force: {physics.named.data.actuator_force['right_finger_act']}")
        # print(f"[DEBUG] Left Gripper Force: {physics.named.data.actuator_force['left_finger_act']}")
        current_reward = self.get_reward(physics)
        print(f"[DEBUG] Step: {self._step_count}, Reward: {current_reward}")
        # for i in range(len(physics.model.id2name)):
        #     print(f"Geom {i}: {physics.model.id2name[i]}")

        # body_name = "ee_finger_l1"  # 원하는 body 이름
        # body_id = physics.model.name2id(body_name, "body")  # body ID 가져오기

        # # 해당 body에 속한 geom들의 ID를 찾기
        # geom_ids = [i for i in range(len(physics.model.geom_bodyid)) if physics.model.geom_bodyid[i] == body_id]

        # # geom ID를 geom 이름으로 변환
        # geom_names = [physics.model.id2name(i, "geom") for i in geom_ids]

        # print(f"Body '{body_name}'에 속한 geom들: {geom_names}")
   

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """
        관측값 반환:
        - 팔 조인트 상태
        - 그리퍼 상태
        - 바퀴 속도
        - 카메라 이미지
        """
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)  # ✅ 'env_state' 추가
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')

        # ✅ 'mocap_pose' 추가
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[3], physics.data.mocap_quat[3]]).copy()
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[4], physics.data.mocap_quat[4]]).copy()

        return obs

    @staticmethod
    def get_env_state(physics):
        # ✅ 박스의 pose (위치 + 회전) 가져오기
        box_id = physics.model.name2id('box', 'body')
        box_pose = physics.data.xpos[box_id]  # (x, y, z, qx, qy, qz, qw)
        # print('box_id:', box_id)
        print('box_pose:', box_pose)
        return box_pose


class TransferCubeTask(RBY1Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self._step_count = 0

    def get_reward(self, physics):
        """
        보상 계산:
        - 오른손이 박스를 잡으면 1점
        - 박스를 들어올리면 2점
        - 왼손이 박스를 잡으면 3점
        - 왼손이 박스를 들면 4점
        """
        # all_contact_pairs = []
        # for i_contact in range(physics.data.ncon):
        #     id_geom_1 = physics.data.contact[i_contact].geom1
        #     id_geom_2 = physics.data.contact[i_contact].geom2
        #     name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
        #     name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
        #     contact_pair = (name_geom_1, name_geom_2)
        #     all_contact_pairs.append(contact_pair)

        touch_left_gripper = is_contacted(physics, "red_box", "ee_finger_l2")
        touch_right_gripper = is_contacted(physics, "red_box", "ee_finger_r2")
        touch_table = is_contacted(physics, "red_box", "tabletop")
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:
            reward = 2
        if touch_left_gripper:
            reward = 3
        if touch_left_gripper and not touch_table:
            reward = 4
        # if not touch_table:
        #     reward = 5
        return reward

    def initialize_episode(self, physics):
        with physics.reset_context():
            self._step_count = 0
            # physics.data.ctrl[:] = 0
            # ✅ 모든 ctrl 0으로 초기화
            np.copyto(physics.data.ctrl, np.zeros_like(physics.data.ctrl))
            # physics.data.qpos[:] = physics.model.qpos0
            
            # ✅ qpos 초기화 (로봇 관절 상태)
            physics.named.data.qpos['right_arm_0'] = -45 * D2R
            physics.named.data.qpos['right_arm_1'] = -45 * D2R
            physics.named.data.qpos['right_arm_2'] = 30  * D2R
            physics.named.data.qpos['right_arm_3'] = -45 * D2R
            physics.named.data.qpos['right_arm_4'] = 20  * D2R
            physics.named.data.qpos['right_arm_5'] = -20 * D2R
            physics.named.data.qpos['right_arm_6'] = 0   * D2R

            physics.named.data.qpos['left_arm_0'] = -45 * D2R
            physics.named.data.qpos['left_arm_1'] = 45  * D2R
            physics.named.data.qpos['left_arm_2'] = -30 * D2R
            physics.named.data.qpos['left_arm_3'] = -45 * D2R
            physics.named.data.qpos['left_arm_4'] = -20 * D2R
            physics.named.data.qpos['left_arm_5'] = -20 * D2R
            physics.named.data.qpos['left_arm_6'] = 0   * D2R

            # # ✅ mocap 초기 위치 강제 설정
            physics.data.mocap_pos[3] = np.array([0.3, 0.2, 1.2])  # 오른손 mocap 위치
            physics.data.mocap_quat[3] = np.array([1, 0, 0, 0])  # 기본 quaternion

            physics.data.mocap_pos[4] = np.array([-0.3, 0.2, 1.2])  # 왼손 mocap 위치
            physics.data.mocap_quat[4] = np.array([1, 0, 0, 0])

            # ✅ 박스 위치 설정
            # assert BOX_POSE[0] is not None
            # box_pose = BOX_POSE[0]
            if BOX_POSE[0] is not None:
                physics.named.data.qpos['red_box_joint'][:7] = BOX_POSE[0]
            # physics.named.data.qvel['red_box_joint'][:] = 0
            print("[DEBUG] Gravity:", physics.model.opt.gravity)
            print("[DEBUG] Initial box velocity:", physics.named.data.qvel['red_box_joint'])


            # ✅ 초기 박스-테이블 접촉 여부 확인
            box_touching_table = is_contacted(physics, "red_box", "tabletop")

            print(f"[DEBUG] 박스 초기 접촉 상태: {'✅ 테이블 위' if box_touching_table else '❌ 공중에 떠 있음'}")

        super().initialize_episode(physics)
        # print("[DEBUG] 🟢 initialize_episode (mocap pos 설정 완료)")
