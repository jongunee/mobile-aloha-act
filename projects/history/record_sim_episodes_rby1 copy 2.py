# record_sim_episodes_rby1.py
import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
from rby1_sdk import *

from constants_rby1 import SIM_TASK_CONFIGS
from sim_env_rby1 import make_sim_env
from scripted_policy_rby1 import HardwarePolicy

import IPython
e = IPython.embed

# 전역 변수: 하드웨어 정책 객체
global_policy = None
D2R = np.pi / 180  # Degree to Radian conversion factor
MINIMUM_TIME = 2.5

def cb(rs):
    """
    rby1_sdk의 콜백함수 (실제 로봇 상태 수신)
    rs.position: 길이=20 (waist6 + right_arm7 + left_arm7)
    """
    global global_policy
    if global_policy is not None:
        global_policy.update_joint_pos(rs.position)

def example_joint_position_command_1(robot):
    print("joint position command example 1")

    # Initialize joint positions
    q_joint_waist = np.zeros(6)
    q_joint_right_arm = np.zeros(7)
    q_joint_left_arm = np.zeros(7)

    # Set specific joint positions
    q_joint_right_arm[1] = -90 * D2R
    q_joint_left_arm[1] = 90 * D2R

    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder().set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_torso_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_waist)
            )
            .set_right_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_right_arm)
            )
            .set_left_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(MINIMUM_TIME)
                .set_position(q_joint_left_arm)
            )
        )
    )

    rv = robot.send_command(rc, 10).get()

    if rv.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Error: Failed to conduct demo motion.")
        return 1

    return 0

def main(args):
    """
    1) rby1 SDK로 실제 로봇을 움직인다 (example_joint_position_command_1).
    2) 콜백(cb)에서 로봇 관절값을 HardwarePolicy에 전달.
    3) DM Control 시뮬레이션에 step(action)하여 기록.
    """
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    address = args['address']

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    # -------------------
    # 0) rby1 SDK 연결

    robot = create_robot_a(address)
    if not robot.connect():
        print("Error: Unable to connect robot.")
        return

    # 콜백 등록
    robot.start_state_update(cb, 0.1)
    # 전원, 서보온, 컨트롤매니저 등 준비 (간략화)
    # ...
    # 실제 로봇을 example_joint_position_command_1로 움직임
    rv = example_joint_position_command_1(robot)
    if rv != 0:
        print("Failed to run example_joint_position_command_1")
        return
    # -------------------

    # -------------------
    # 1) HardwarePolicy 생성
    global global_policy
    global_policy = HardwarePolicy()
    # -------------------

    success = []
    for episode_idx in range(num_episodes):
        print(f'episode_idx={episode_idx}')
        # 2) 시뮬레이션 환경 생성
        env = make_sim_env(task_name)
        ts = env.reset()

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][camera_names[0]])
            plt.ion()

        # 시뮬레이션 진행
        episode = [ts]
        for step in range(episode_len):
            # HardwarePolicy에서 실제로봇 관절 → sim action
            action = global_policy(ts)
            ts = env.step(action)
            episode.append(ts)

            if onscreen_render:
                plt_img.set_data(ts.observation['images'][camera_names[0]])
                plt.pause(0.002)
        plt.close()

        # episode_return 등 계산 (데모라 보상=0)
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        print(f'Rollout: episode_return={episode_return}')

        # qpos trajectory 기록
        joint_traj = [ts.observation['qpos'] for ts in episode]

        # 3) replay (옵션 - 필요 시)
        #   replay해도 어차피 HW와 똑같이 움직인 trajectory를 재현
        env_replay = make_sim_env(task_name)
        ts_replay = env_replay.reset()
        episode_replay = [ts_replay]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts_replay.observation['images'][camera_names[0]])
            plt.ion()
        for t in range(len(joint_traj)):
            action = joint_traj[t]
            ts_replay = env_replay.step(action)
            episode_replay.append(ts_replay)
            if onscreen_render:
                plt_img.set_data(ts_replay.observation['images'][camera_names[0]])
                plt.pause(0.02)
        plt.close()

        # 4) 데이터 HDF5 저장
        #   - 기존 record_sim_episodes_rby1.py 로직 그대로
        max_timesteps = len(joint_traj)
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # truncate matching length
        traj_copy = joint_traj.copy()
        replay_copy = episode_replay.copy()

        replay_copy.pop()  # step수 맞추기
        while traj_copy:
            action = traj_copy.pop(0)
            ts_local = replay_copy.pop(0)
            data_dict['/observations/qpos'].append(ts_local.observation['qpos'])
            data_dict['/observations/qvel'].append(ts_local.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts_local.observation['images'][cam_name])

        qpos_dim = len(data_dict['/observations/qpos'][0])
        qvel_dim = len(data_dict['/observations/qvel'][0])
        action_dim = len(data_dict['/action'][0])

        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image_group = obs.create_group('images')
            for cam_name in camera_names:
                image_group.create_dataset(cam_name, (max_timesteps, 480, 640, 3),
                                           dtype='uint8', chunks=(1, 480, 640, 3))
            qpos_ds = obs.create_dataset('qpos', (max_timesteps, qpos_dim))
            qvel_ds = obs.create_dataset('qvel', (max_timesteps, qvel_dim))
            action_ds = root.create_dataset('action', (max_timesteps, action_dim))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time()-t0:.1f} secs\n')

        success.append(1)

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help='task_name (e.g., sim_rby1_demo)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset saving dir')
    parser.add_argument('--num_episodes', type=int, default=1, help='num_episodes')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--address', type=str, required=True, help="Robot address (e.g. 192.168.0.1)")
    main(vars(parser.parse_args()))
