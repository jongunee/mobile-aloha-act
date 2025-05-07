import pathlib

DT = 0.02
FPS = 50
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/models/rainbow_robotics_rby1-training/'

SIM_TASK_CONFIGS = {
  "sim_transfer_cube": {
    # "dataset_dir": "/mnt/storage/jwpark/mobile_aloha/datasets/rby1_transfer_random_cube_only_success",
    "dataset_dir": "/mnt/storage/jwpark/mobile_aloha/datasets/rby1_transfer_cam_top_open_start_no_noise/",
    "episode_len": 400,
    "camera_names": ["top"],
    "train_ratio": 0.99,
    "name_filter": lambda n: True,
    "stats_dir": None,
    "sample_weights": None
  }
}
