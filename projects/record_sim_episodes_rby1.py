import os
import numpy as np
import time
import argparse
import h5py
import matplotlib.pyplot as plt
import logging
from sim_env_rby1 import make_sim_env, BOX_POSE, random_box_pose
from scripted_policy_rby1 import PickAndTransferPolicy

def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거 (중복 방지)
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # 파일 핸들러 추가
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def main(args):
    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # Simulation 환경 생성성
    env = make_sim_env(task_name)
    policy = PickAndTransferPolicy(inject_noise=False)

    # 시각화 설정시시
    if onscreen_render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax.axis('off')
        plt.show()

    success = []
    log_path = os.path.join(dataset_dir, 'result.log')
    setup_logger(log_path)
    for episode_idx in range(num_episodes):
        print(f"\n=== Episode {episode_idx} 시작 ===")
        logging.info(f"Episode {episode_idx} 시작")
        
        # 박스 초기 위치 설정
        BOX_POSE[0] = random_box_pose()
        # BOX_POSE[0] = np.array([0.2, 0.3, 0.94, 1, 0, 0, 0])

        ts = env.reset()
        box_pos = np.array(ts.observation['env_state'])
        policy.generate_trajectory(ts)
        print("[DEBUG] 박스 위치:", box_pos)
        logging.info(f"초기 박스 위치: {box_pos}")

        # 데이터로 저장할 요소들
        data_dict = {
            'observations/qpos': [],
            'observations/qvel': [],
            'action': [],
        }
        camera_names = ['top']
        for cam_name in camera_names:
            data_dict[f'observations/images/{cam_name}'] = []

        max_reward = env.task.max_reward
        rewards = []

        for t in range(400):
            action = policy(ts)
            ts = env.step(action)

            data_dict['observations/qpos'].append(ts.observation['qpos'])
            data_dict['observations/qvel'].append(ts.observation['qvel'])
            data_dict['action'].append(action)

            for cam_name in camera_names:
                data_dict[f'observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            if onscreen_render:
                img = ts.observation['images']['top']
                im.set_data(img)
                ax.set_title(f"Episode {episode_idx} - Step {t}")
                plt.pause(0.02)

            current_reward = env.task.get_reward(env.physics)
            rewards.append(current_reward)
            print(f"[DEBUG] Step {t}, Reward: {current_reward}")

        episode_return = np.sum(rewards)
        episode_max_reward = np.max(rewards)
        if episode_max_reward == max_reward:
            print(f"Episode {episode_idx}: 성공 (Return: {episode_return})")
            success.append(1)
            logging.info(f"[Episode {episode_idx}] SUCCESS - Return: {episode_return} / Max: {episode_max_reward}")
        else:
            print(f"Episode {episode_idx}: 실패")
            success.append(0)
            logging.info(f"[Episode {episode_idx}] FAIL    - Return: {episode_return} / Max: {episode_max_reward}")

        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('observations/qpos', data=np.array(data_dict['observations/qpos']))
            f.create_dataset('observations/qvel', data=np.array(data_dict['observations/qvel']))
            f.create_dataset('action', data=np.array(data_dict['action']))
            for cam_name in camera_names:
                f.create_dataset(f'observations/images/{cam_name}', data=np.array(data_dict[f'observations/images/{cam_name}']), dtype='uint8')

        print(f"Saved: {dataset_path}")

    total_success = np.sum(success)
    success_rate = np.mean(success) * 100
    print(f"\n총 성공률: {total_success} / {len(success)} = {success_rate:.2f}%")
    logging.info(f"총 성공률: {total_success} / {len(success)} = {success_rate:.2f}%")

    if onscreen_render:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube', help='Task name')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/storage/jwpark/mobile_aloha/datasets/rby1_transfer_cam_top_open_start_no_noise', help='Dataset saving directory')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--onscreen_render', action='store_true', default=0, help='Enable on-screen rendering')

    main(vars(parser.parse_args()))
