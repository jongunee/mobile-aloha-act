import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms
import cv2 

from constants_rby1 import FPS
from utils_rby1 import load_data
from utils_rby1 import sample_box_pose
from utils_rby1 import compute_dict_mean, set_seed
from policy_rby1 import ACTPolicy
from detr.models.latent_model import Latent_Model_Transformer
from sim_env_rby1 import BOX_POSE, make_sim_env

import IPython
e = IPython.embed

def main(args):
    import os
    import pickle

    set_seed(args['seed'])
    is_eval = args['eval']

    # 기본 config 세팅
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    onscreen_render = args['onscreen_render']
    batch_size = args['batch_size']
    chunk_size = args['chunk_size']
    camera_names = ['top']  # 기본 카메라

    # 공통 policy config
    policy_config = {
        'lr': args['lr'],
        'num_queries': chunk_size,
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'position_embedding': 'sine',
        'dilation': False,
        'dropout': 0.1,
        'nheads': 8,
        'enc_layers': 4,
        'dec_layers': 7,
        'camera_names': camera_names,
        'vq': args['use_vq'],
        'vq_class': args['vq_class'],
        'vq_dim': args['vq_dim'],
        'action_dim': 16,
        'no_encoder': args['no_encoder'],
        'masks': False,
        'pre_norm': False,
    }

    # 공통 config
    config = {
        'ckpt_dir': ckpt_dir,
        'policy_class': policy_class,
        'task_name': task_name,
        'seed': args['seed'],
        'state_dim': 42,
        'lr': args['lr'],
        'policy_config': policy_config,
        'episode_len': 400,
        'onscreen_render': onscreen_render,
        'camera_names': camera_names,
        'real_robot': False,
        'load_pretrain': args['load_pretrain'],
    }

    if is_eval:
        # 평가 모드 (학습 X)
        eval_bc(config, ckpt_name='policy_best.ckpt', save_episode=False, num_rollouts=1)

    else:
        # 학습 모드
        from constants_rby1 import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
        dataset_dir = task_config['dataset_dir']
        stats_dir = task_config.get('stats_dir', None)
        sample_weights = task_config.get('sample_weights', None)
        train_ratio = task_config.get('train_ratio', 0.99)
        name_filter = task_config.get('name_filter', lambda n: True)
        if isinstance(name_filter, str):
            name_filter = eval(name_filter)

        os.makedirs(ckpt_dir, exist_ok=True)
        config_path = os.path.join(ckpt_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        wandb.init(project="rby1-imitate", reinit=True, name=os.path.basename(ckpt_dir))
        wandb.config.update(config)

        train_loader, val_loader, stats, _ = load_data(
            dataset_dir,
            name_filter,
            camera_names,
            batch_size,
            batch_size,
            chunk_size,
            args['skip_mirrored_data'],
            args['load_pretrain'],
            policy_class,
            stats_dir_l=stats_dir,
            sample_weights=sample_weights,
            train_ratio=train_ratio
        )

        config['num_steps'] = args['num_steps']
        config['eval_every'] = args['eval_every']
        config['validate_every'] = args['validate_every']
        config['save_every'] = args['save_every']
        config['resume_ckpt_path'] = args['resume_ckpt_path']

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        best_ckpt = train_bc(train_loader, val_loader, config)
        best_step, min_val_loss, best_state_dict = best_ckpt
        torch.save(best_state_dict, os.path.join(ckpt_dir, f'policy_best.ckpt'))
        print(f'Best val loss: {min_val_loss:.6f} @ step {best_step}')

        wandb.finish()



def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10):
    import matplotlib.pyplot as plt

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    onscreen_cam = 'top'  # 기본 카메라
    vq = config['policy_config']['vq']

    # load policy
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()

    if vq:
        from detr.models.latent_model import Latent_Model_Transformer
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')

    # load stats
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # create environment
    from sim_env_rby1 import make_sim_env, BOX_POSE
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    FPS = 50
    DT = 1 / FPS

    episode_returns = []
    highest_rewards = []

    for rollout_id in range(num_rollouts):
        # BOX_POSE[0] = np.array([0.2, 0.3, 0.94, 1, 0, 0, 0])
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()

        if onscreen_render:
            fig, ax = plt.subplots(figsize=(8, 6))
            img = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            plt_img = ax.imshow(img)
            plt.axis('off')
            plt.ion()
            plt.show()

        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                if t % query_frequency == 0:
                    curr_images = []
                    for cam_name in camera_names:
                        image = obs['images'][cam_name]
                        image = rearrange(image, 'h w c -> c h w')
                        curr_images.append(image)
                    curr_image = torch.from_numpy(np.stack(curr_images) / 255.0).float().cuda().unsqueeze(0)

                    if policy_class == 'ACT':
                        if vq:
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                        else:
                            all_actions = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                if onscreen_render:
                    frame = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(frame)
                    plt.pause(DT)

                raw_action = all_actions[:, t % query_frequency].squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                ts = env.step(action)
                rewards.append(ts.reward)

            if onscreen_render:
                plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        highest_reward = np.max(rewards)
        episode_returns.append(episode_return)
        highest_rewards.append(highest_reward)
        print(f'Rollout {rollout_id}: Return = {episode_return}, Highest Reward = {highest_reward}')

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    print(f'\nSuccess rate: {success_rate}, Average return: {avg_return}\n')
    return success_rate, avg_return



# --- train_bc, forward_pass, repeater는 기존 imitate_episodes.py와 동일하게 아래에 이어 붙이면 됩니다 ---
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        if step > 0 and step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            if config['onscreen_render']:
                print('[Rendering validation rollout]')
                tmp_ckpt_path = os.path.join(ckpt_dir, '__temp_val_ckpt.ckpt')
                torch.save(policy.serialize(), tmp_ckpt_path)
                eval_bc({**config, 'onscreen_render': True}, ckpt_name='__temp_val_ckpt.ckpt', save_episode=False, num_rollouts=10)
                
        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=1)
            wandb.log({'success': success}, step=step)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='/mnt/storage/jwpark/mobile_aloha/ckpt/rby1_transfer_cam_top_open_start_no_noise')
    parser.add_argument('--policy_class', type=str, default='ACT')
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--resume_ckpt_path', type=str, default=None)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=8)

    # ACT-specific
    parser.add_argument('--kl_weight', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', type=int, default=128)
    parser.add_argument('--vq_dim', type=int, default=32)
    parser.add_argument('--no_encoder', action='store_true')

    args = parser.parse_args()
    main(vars(args))
