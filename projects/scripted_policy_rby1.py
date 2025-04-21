import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class PickAndTransferPolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.left_trajectory = []
        self.right_trajectory = []
        self.current_step = 0

        # 보간을 위한 현재/다음 waypoint
        self.curr_left_waypoint = None
        self.curr_right_waypoint = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t0 = curr_waypoint["t"]
        t1 = next_waypoint["t"]
        if t1 == t0:
            alpha = 1.0
        else:
            alpha = (t - t0) / float(t1 - t0)
        alpha = np.clip(alpha, 0.0, 1.0)

        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']

        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']

        # ✅ Quaternion 보간을 Slerp로 변경
        key_rots = R.from_quat([curr_quat, next_quat])
        slerp = Slerp([0, 1], key_rots)
        quat = slerp(alpha).as_quat()

        # ✅ XYZ는 기존 선형 보간 유지
        xyz = curr_xyz + (next_xyz - curr_xyz) * alpha
        gripper = curr_grip + (next_grip - curr_grip) * alpha

        return xyz, quat, gripper

    def __call__(self, ts):
        """
        매 시뮬레이션 스텝마다 액션(16차원)을 반환
        - 오른손(xyz + quat + gripper) 8차원
        - 왼손(xyz + quat + gripper) 8차원
        """
        # 첫 스텝이면 궤적 생성
        if self.current_step == 0:
            self.generate_trajectory(ts)

            # 맨 처음 waypoint를 현재 waypoint로 설정
            if self.left_trajectory:
                self.curr_left_waypoint = self.left_trajectory.pop(0)
            if self.right_trajectory:
                self.curr_right_waypoint = self.right_trajectory.pop(0)

        # 남아 있는 다음 왼손 waypoint
        if self.left_trajectory and (self.left_trajectory[0]["t"] <= self.current_step):
            # 만약 다음 waypoint의 t가 현재 step 이하라면, 이미 지나친 것이므로 즉시 pop
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0] if self.left_trajectory else self.curr_left_waypoint

        # 남아 있는 다음 오른손 waypoint
        if self.right_trajectory and (self.right_trajectory[0]["t"] <= self.current_step):
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0] if self.right_trajectory else self.curr_right_waypoint

        # 왼손/오른손 각각 보간
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.current_step
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.current_step
        )

        # 노이즈 적용 (선택)
        if self.inject_noise:
            scale = 0.001
            left_xyz += np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz += np.random.uniform(-scale, scale, right_xyz.shape)

        # 액션은 16차원
        action = np.zeros(16)
        # 오른손 (xyz + quat + gripper)
        action[:3] = right_xyz
        action[3:7] = right_quat
        action[14] = right_gripper
        # 왼손 (xyz + quat + gripper)
        action[7:10] = left_xyz
        action[10:14] = left_quat
        action[15] = left_gripper

        self.current_step += 1
        return action

    def generate_trajectory(self, ts_first):
        self.current_step = 0
        """
        초기 위치 기반 엔드이펙터 궤적(waypoints) 설정
        """
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        print("box_info: ", box_info)
        
        meet_xyz = np.array([0, 0.3, 1.3])  # 두 손이 만나는 위치

        # 오른손 회전 보정
        gripper_pick_quat = R.from_quat(init_mocap_pose_right[3:]) * R.from_euler('y', 60, degrees=True)
        gripper_pick_quat = gripper_pick_quat.as_quat()

        # 왼손 회전 보정 (x축 90도)
        meet_left_quat = R.from_quat(init_mocap_pose_left[3:]) * R.from_euler('y', -90, degrees=True) * R.from_euler('z', 90, degrees=True)
        meet_left_quat = meet_left_quat.as_quat()
        print(f"[DEBUG] Initial Left Hand Quaternion: {init_mocap_pose_left[3:]}")
        print(f"[DEBUG] meet_left_quat: {meet_left_quat}")

        # Left Trajectory
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, -0.07, -0.02]), "quat": meet_left_quat, "gripper": 10},
            {"t": 260, "xyz": meet_xyz + np.array([-0.05, -0.07, -0.02]), "quat": meet_left_quat, "gripper": 10},
            {"t": 300, "xyz": meet_xyz + np.array([-0.05, -0.07, -0.02]), "quat": meet_left_quat, "gripper": -10},
            {"t": 320, "xyz": meet_xyz + np.array([-0.05, -0.07, -0.02]), "quat": meet_left_quat, "gripper": -10},
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, -0.07, -0.02]), "quat": meet_left_quat, "gripper": -10},
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, -0.07, -0.02]), "quat": meet_left_quat, "gripper": -10},
        ]

        # Right Trajectory
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
            {"t": 30, "xyz": box_xyz + np.array([0.06, 0, 0.1]), "quat": gripper_pick_quat, "gripper": 10},
            {"t": 60, "xyz": box_xyz + np.array([0.06, 0, 0.05]), "quat": gripper_pick_quat, "gripper": 10},
            {"t": 80, "xyz": box_xyz + np.array([0.06, 0, 0.05]), "quat": gripper_pick_quat, "gripper": -10},
            {"t": 120, "xyz": box_xyz + np.array([0.06, 0, 0.05]), "quat": gripper_pick_quat, "gripper": -10},
            {"t": 220, "xyz": meet_xyz + np.array([0.05, 0, 0.05]), "quat": gripper_pick_quat, "gripper": -10}, # approach meet position
            {"t": 250, "xyz": meet_xyz + np.array([-0.02, 0, 0.05]), "quat": gripper_pick_quat, "gripper":-10},
            {"t": 300, "xyz": meet_xyz + np.array([-0.02, 0, 0.05]), "quat": gripper_pick_quat, "gripper":-10},
            {"t": 340, "xyz": meet_xyz + np.array([-0.02, 0, 0.05]), "quat": gripper_pick_quat, "gripper": 10},
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat, "gripper": 10},
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat, "gripper": 10},
        ]
