import torch
import numpy as np
import time
import argparse
import pickle
from policy_rby1 import ACTPolicy
from rby1_sdk import *

D2R = np.pi / 180
R2D = 180 / np.pi

latest_state = None

def cb(rs):
    global latest_state
    latest_state = rs

# === 정책 불러오기 ===
def load_policy(ckpt_path, config):
    policy = ACTPolicy(config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    return policy

# === 전처리 / 후처리 ===
def preprocess_qpos(qpos, stats):
    padded_qpos = np.zeros_like(stats['qpos_mean'])
    padded_qpos[:len(qpos)] = qpos
    return (padded_qpos - stats['qpos_mean']) / stats['qpos_std']

def postprocess_action(action, stats):
    return action * stats['action_std'] + stats['action_mean']

# === 이미지 전처리 ===
def preprocess_image(image_np):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # [C, H, W]
    return image_tensor.unsqueeze(0).cuda()  # [1, 3, H, W]

# === 로봇 제어 명령 전송 ===
def send_robot_command(robot, target_qpos, base_velocity):
    rc = RobotCommandBuilder().set_command(
        ComponentBasedCommandBuilder()
        .set_body_command(
            BodyComponentBasedCommandBuilder()
            .set_right_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(2.0)
                .set_position(target_qpos[:7])
            )
            .set_left_arm_command(
                JointPositionCommandBuilder()
                .set_minimum_time(2.0)
                .set_position(target_qpos[7:14])
            )
        )
        .set_mobility_command(
            MobilityCommandBuilder()
            .set_command(
                JointVelocityCommandBuilder()
                .set_velocity(base_velocity.tolist())
                .set_command_header(CommandHeaderBuilder().set_control_hold_time(2.0))
            )
        )
    )

    # 정확히 완료될 때까지 기다림
    handler = robot.send_command(rc)
    result = handler.get()  # 기다림
    if result.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("\u274c Command failed!")
        return False
    return True



# === 메인 함수 ===
def main(args):
    global latest_state

    robot = create_robot_a(args.address)
    assert robot.connect(), "\u274c Failed to connect to robot."
    robot.start_state_update(cb, 0.05)
    print("\u2705 Connected to robot.")

    robot.power_on(args.device)
    robot.servo_on(args.servo)
    robot.enable_control_manager()

    with open(args.stats_path, 'rb') as f:
        stats = pickle.load(f)
    policy = load_policy(args.ckpt_path, args.policy_config)
    print("\u2705 Policy loaded.")

    time.sleep(1.0)

    num_cams = len(args.policy_config['camera_names'])

    for t in range(args.num_steps):
        if latest_state is None:
            print("대기 중...")
            time.sleep(0.1)
            continue

        qpos = latest_state.position
        qpos_input = torch.from_numpy(preprocess_qpos(qpos, stats)).float().unsqueeze(0).cuda()

        # 더미 이미지: [1, N, 3, 224, 224]
        dummy_image = torch.zeros((1, num_cams, 3, 224, 224)).cuda()

        with torch.inference_mode():
            action = policy(qpos_input, dummy_image)
            action = action.squeeze(0)[-1]

        action = postprocess_action(action.cpu().numpy(), stats)
        clip_limit_qpos = 0.015
        clip_limit_base = 0.1
        action = np.clip(action, -clip_limit_qpos, clip_limit_qpos)

        target_qpos = action[:-2].astype(np.float64).reshape(-1, 1)
        base_velocity = action[-2:].astype(np.float64).reshape(-1)
        print(f"[{t}] ▶ Sending qpos: {target_qpos.flatten()}, base: {base_velocity}")


        print(f"[{t}] 실행 중... base: {base_velocity}, right arm: {target_qpos[:7]}")
        send_robot_command(robot, target_qpos, base_velocity)

        time.sleep(2.5)

    print("\u2705 Policy 실행 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default="localhost:50051")
    parser.add_argument('--device', type=str, default=".*")
    parser.add_argument('--servo', type=str, default=".*")
    parser.add_argument('--ckpt_path', type=str,  help="학습된 정책 경로 (.ckpt)", default='/mnt/storage/jwpark/mobile_aloha/ckpt/rby1_transfer_cam_top_open_start_no_noise/policy_best.ckpt')
    parser.add_argument('--stats_path', type=str,  help="정규화 통계 파일 (.pkl)", default='/mnt/storage/jwpark/mobile_aloha/ckpt/rby1_transfer_cam_top_open_start_no_noise/dataset_stats.pkl')
    parser.add_argument('--num_steps', type=int, default=100)
    args = parser.parse_args()

    args.policy_config = {
        'lr': 1e-4,
        'num_queries': 8,
        'kl_weight': 1,
        'hidden_dim': 256,
        'dim_feedforward': 1024,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': ['top'],
        'vq': False,
        'vq_class': 128,
        'vq_dim': 32,
        'action_dim': 16,
        'no_encoder': False,
        'masks': False,
        'pre_norm': False,
        'position_embedding': 'sine',
        'dilation': False,
        'dropout': 0.1
    }

    main(args)
