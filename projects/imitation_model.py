import torch
import numpy as np
import time
import argparse
import pickle
import requests
import cv2
from policy_rby1 import ACTPolicy
from rby1_sdk import *

D2R = np.pi / 180
R2D = 180 / np.pi
latest_state = None

def cb(rs):
    global latest_state
    latest_state = rs

def load_policy(ckpt_path, config):
    policy = ACTPolicy(config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    return policy

def preprocess_qpos(qpos, stats):
    padded_qpos = np.zeros_like(stats['qpos_mean'])
    padded_qpos[:len(qpos)] = qpos
    return (padded_qpos - stats['qpos_mean']) / stats['qpos_std']

def postprocess_action(action, stats):
    return action * stats['action_std'] + stats['action_mean']

def preprocess_image(image_np):
    image_np = cv2.resize(image_np, (224, 224))
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    return image_tensor.unsqueeze(0).cuda()

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

    handler = robot.send_command(rc)
    result = handler.get()
    if result.finish_code != RobotCommandFeedback.FinishCode.Ok:
        print("Command failed!")
        return False
    return True

def get_camera_image():
    try:
        response = requests.get("http://localhost:8999/top.jpg", timeout=1.0)
        if response.status_code == 200:
            image_arr = np.frombuffer(response.content, dtype=np.uint8)
            image_np = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
            return image_np
        else:
            return None
    except Exception as e:
        print(f"[Camera Error] {e}")
        return None

def main(args):
    global latest_state

    robot = create_robot_a(args.address)
    assert robot.connect(), "Failed to connect to robot."
    robot.start_state_update(cb, 0.05)
    print("Connected to robot.")

    robot.power_on(args.device)
    robot.servo_on(args.servo)
    robot.enable_control_manager()

    with open(args.stats_path, 'rb') as f:
        stats = pickle.load(f)
    policy = load_policy(args.ckpt_path, args.policy_config)
    print("Policy loaded.")

    time.sleep(1.0)

    for t in range(args.num_steps):
        if latest_state is None:
            print("대기 중...")
            time.sleep(0.1)
            continue

        qpos = latest_state.position
        qpos_input = torch.from_numpy(preprocess_qpos(qpos, stats)).float().unsqueeze(0).cuda()

        image_np = get_camera_image()
        if image_np is None:
            print("❗ 카메라 이미지 수신 실패. step skip.")
            continue

        image_tensor = preprocess_image(image_np).unsqueeze(1)  # [1, 1, 3, 224, 224]

        with torch.inference_mode():
            action = policy(qpos_input, image_tensor)
            action = action.squeeze(0)[-1]

        action = postprocess_action(action.cpu().numpy(), stats)
        clip_limit_qpos = 0.001
        clip_limit_base = 0.01
        action = np.clip(action, -clip_limit_qpos, clip_limit_qpos)

        target_qpos = action[:-2].astype(np.float64).reshape(-1, 1)
        base_velocity = action[-2:].astype(np.float64).reshape(-1)

        print(f"[{t}] ▶ Sending qpos: {target_qpos.flatten()}, base: {base_velocity}")
        send_robot_command(robot, target_qpos, base_velocity)
        time.sleep(2.5)

    print("Policy 실행 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default="localhost:50051")
    parser.add_argument('--device', type=str, default=".*")
    parser.add_argument('--servo', type=str, default=".*")
    parser.add_argument('--ckpt_path', type=str, default='/mnt/storage/jwpark/mobile_aloha/ckpt/rby1_transfer_cam_top_open_start_no_noise/policy_best.ckpt')
    parser.add_argument('--stats_path', type=str, default='/mnt/storage/jwpark/mobile_aloha/ckpt/rby1_transfer_cam_top_open_start_no_noise/dataset_stats.pkl')
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
