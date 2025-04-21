import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants_rby1 import SIM_TASK_CONFIGS
# from ee_sim_env_rby1 import make_ee_sim_env
from sim_env_rby1 import make_sim_env
# 수정: rby1 데모용 정책을 사용하도록 함
from scripted_policy_rby1 import PickAndTransferPolicy

import IPython
e = IPython.embed

def main(args):
    """
    시뮬레이션 데모 데이터를 생성합니다.
    1) 먼저 EE 제어 환경(엔드 이펙터 제어)에서 스크립트 정책(Rby1DemoPolicy)을 rollout하여
       로봇의 joint trajectory와 관측 데이터를 기록합니다.
    2) 기록된 joint trajectory를 관절 제어(sim_env) 환경에서 재현(replay)하여 데이터를 다시 기록하고,  
       HDF5 파일로 저장합니다.
    """
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    # task에 따라 사용할 정책 선택 (rby1 데모 태스크인 경우)
    if task_name == 'sim_rby1_demo':
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError("해당 task는 현재 지원하지 않습니다.")

    success = []
    for episode_idx in range(num_episodes):
        print(f'episode_idx={episode_idx}')
        print('Rollout: EE simulation with scripted policy')
        # ① EE 시뮬레이션 환경 생성 (rby1 데모 환경)
        env = make_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls()
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][camera_names[0]])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][camera_names[0]])
                plt.pause(0.002)
        plt.close()

        # 간단 평가: (여기서는 데모라 보상은 0일 수 있음)
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        print(f'Rollout: episode_return={episode_return}')

        # joint trajectory 기록 (rby1의 경우 별도 그리퍼 보정 없이 qpos 그대로 사용)
        joint_traj = [ts.observation['qpos'] for ts in episode]
        # subtask_info 기록 (여기서는 초기 qpos 전체를 사용)
        subtask_info = episode[0].observation['qpos'].copy()

        del env, episode, policy

        # ② 관절 제어 환경에서 replay (기록된 joint trajectory를 그대로 실행)
        print('Replaying joint commands in sim_env')
        env = make_sim_env(task_name)
        # rby1 데모의 경우 추가 객체가 없으므로 별도 초기 상태 조정 없음
        ts = env.reset()
        episode_replay = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][camera_names[0]])
            plt.ion()
        for t in range(len(joint_traj)):
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][camera_names[0]])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        print(f'Replay: episode_return={episode_return}')
        success.append(1)  # 데모에서는 별도 성공 판정 없이 기록했다고 가정

        # ③ HDF5에 데이터 저장  
        # 관측 데이터 차원은 첫 timestep에서 추출
        max_timesteps = len(joint_traj)
        data_dict = {}
        # 각 관측의 리스트 생성 (qpos, qvel, action)
        data_dict['/observations/qpos'] = []
        data_dict['/observations/qvel'] = []
        data_dict['/action'] = []
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # replay 데이터에서 각 timestep마다 기록
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # 데이터 차원 동적 결정 (첫 timestep 기준)
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
                # 이미지 크기는 (480, 640, 3)로 고정
                image_group.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                           chunks=(1, 480, 640, 3))
            qpos_ds = obs.create_dataset('qpos', (max_timesteps, qpos_dim))
            qvel_ds = obs.create_dataset('qvel', (max_timesteps, qvel_dim))
            action_ds = root.create_dataset('action', (max_timesteps, action_dim))
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving time: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help='task name (e.g., sim_rby1_demo)')
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset saving dir')
    parser.add_argument('--num_episodes', type=int, default=50, help='number of episodes')
    parser.add_argument('--onscreen_render', action='store_true')
    main(vars(parser.parse_args()))
