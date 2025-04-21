import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def sample_box_pose():
    x = np.random.uniform(0.15, 0.25)
    y = np.random.uniform(0.25, 0.35)
    z = 0.94  # 고정
    cube_quat = np.array([1, 0, 0, 0])
    return np.array([x, y, z, *cube_quat])


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename:
                continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    action_mean = all_action_data.mean(dim=0).float()
    action_std = all_action_data.std(dim=0).float().clamp(min=1e-2)
    qpos_mean = all_qpos_data.mean(dim=0).float()
    qpos_std = all_qpos_data.std(dim=0).float().clamp(min=1e-2)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 1e-4
    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": (action_min - eps).numpy(),
        "action_max": (action_max + eps).numpy(),
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": qpos
    }
    return stats, all_episode_len


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


class EpisodicDataset(Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.augment_images = policy_class == 'Diffusion'
        self.transformations = None
        self.__getitem__(0)
        self.is_sim = False

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][start_ts]
                action = root['/action'][()]
                image_dict = {cam: root[f'/observations/images/{cam}'][start_ts] for cam in self.camera_names}

            action_data = action[start_ts:]
            padded_action = np.zeros((self.max_episode_len, action.shape[1]), dtype=np.float32)
            padded_action[:len(action_data)] = action_data
            is_pad = np.zeros(self.max_episode_len)
            is_pad[len(action_data):] = 1
            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            all_cam_images = np.stack([image_dict[cam] for cam in self.camera_names], axis=0)
            image_data = torch.from_numpy(all_cam_images).float().permute(0, 3, 1, 2) / 255.0
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            if self.transformations is None and self.augment_images:
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0]*ratio), int(original_size[1]*ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            if self.policy_class == 'Diffusion':
                action_data = ((action_data - self.norm_stats['action_min']) / (self.norm_stats['action_max'] - self.norm_stats['action_min'])) * 2 - 1
            else:
                action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']

            qpos_data = (qpos_data - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']

        except Exception as e:
            print(f'Error loading {dataset_path} in __getitem__')
            print(e)
            quit()

        return image_data, qpos_data, action_data, is_pad


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99):
    if isinstance(dataset_dir_l, str):
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(paths) for paths in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(n) + num_episodes_cumsum[i] for i, n in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)

    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)

    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif isinstance(stats_dir_l, str):
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(s, skip_mirrored_data) for s in stats_dir_l]))

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=4)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
