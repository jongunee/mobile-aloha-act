import numpy as np
import collections
import os
from constants_rby1 import DT, XML_DIR, START_QPOS_RBY1  # ✅ RBY1 관련 상수
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from utils import sample_box_pose

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for RBY1 robot manipulation, with end-effector control.
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'rby1.xml')  # ✅ RBY1 환경 사용
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)  # ✅ 새로운 Task 클래스 사용
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env


class RBY1EETask(base.Task):  # ✅ 새로운 Task 베이스 클래스
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        """
        - action: 엔드이펙터의 목표 위치 및 회전값을 받음
        - physics.data.mocap_pos[0] (왼팔)
        - physics.data.mocap_pos[1] (오른팔)
        """
        left_xyz = action[:3]
        left_quat = action[3:7]
        right_xyz = action[7:10]
        right_quat = action[10:14]
        gripper_left = action[14]
        gripper_right = action[15]

        # ✅ 엔드이펙터 위치 직접 이동
        np.copyto(physics.data.mocap_pos[0], left_xyz)
        np.copyto(physics.data.mocap_quat[0], left_quat)
        np.copyto(physics.data.mocap_pos[1], right_xyz)
        np.copyto(physics.data.mocap_quat[1], right_quat)

        # ✅ 그리퍼 컨트롤
        np.copyto(physics.data.ctrl, np.array([gripper_left, -gripper_left, gripper_right, -gripper_right]))

    def initialize_episode(self, physics):
        """
        - 초기 팔 위치 및 gripper 초기화
        - 초기 box 위치 설정
        """
        with physics.reset_context():
            physics.named.data.qpos[:len(START_QPOS_RBY1)] = START_QPOS_RBY1  # ✅ 초기 자세 설정
            np.copyto(physics.data.ctrl, np.zeros_like(physics.data.ctrl))  # ✅ 그리퍼 초기화

            # ✅ 초기 엔드이펙터 위치 설정
            np.copyto(physics.data.mocap_pos[0], [-0.3, 0.5, 0.3])  # 왼쪽 팔
            np.copyto(physics.data.mocap_pos[1], [0.3, 0.5, 0.3])   # 오른쪽 팔
            np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
            np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

            # ✅ 박스 위치 랜덤 초기화
            cube_pose = sample_box_pose()
            box_start_idx = physics.model.name2id('red_box_joint', 'joint')
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """
        관측값 반환:
        - 팔 위치 및 속도
        - gripper 상태
        - 카메라 이미지
        """
        obs = collections.OrderedDict()
        obs['qpos'] = physics.data.qpos.copy()
        obs['qvel'] = physics.data.qvel.copy()
        obs['images'] = {
            'top': physics.render(height=480, width=640, camera_id='top')
        }
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        return obs


class TransferCubeEETask(RBY1EETask):  # ✅ TransferCubeTask 정의
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def get_reward(self, physics):
        """
        보상 계산:
        - 오른손이 박스를 잡으면 1점
        - 박스를 들어올리면 2점
        - 왼손이 박스를 잡으면 3점
        - 왼손이 박스를 들면 4점
        """
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "ee_finger_l1") in all_contact_pairs
        touch_right_gripper = ("red_box", "ee_finger_r1") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:
            reward = 2
        if touch_left_gripper:
            reward = 3
        if touch_left_gripper and not touch_table:
            reward = 4
        return reward
