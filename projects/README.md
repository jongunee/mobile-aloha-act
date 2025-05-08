# RB-Y1 모방학습

### 파일 구조

- `constants_rby1.py`: 시뮬레이션 데이터셋 경로, 모델 파일 경로로 등의 설정 포함
- `imitate_episodes_rby1.py`: 모방학습 메인 스크립트로 데이터 로딩, 학습, 검증, 체크포인트 저장 등 포함
- `policy_rby1.py`: 모방학습용 ACT 모델 정의
- `record_sim_episodes_rby1.py`: 시뮬레이터에서 스크립트 정책을 이용해 HDF5 형식으로 학습용 데이터 수집
- `scripted_policy_rby1.py`: 휴머노이드 양팔의 궤적을 정의한 정책
- `sim_env_rby1.py`: MuJoCo 기반의 RB-Y1 시뮬레이션 환경 정의해해 로봇 초기화, 액션 적용, 보상 계산, 관측 처리 등 포함
- `utils_rby1.py`: 데이터 로딩, 정규화, 배치 샘플링, 통계 계산 등의 유틸 함수 모음
- `imitation_model.py`: 학습시킨 모델을 적용

### 데이터 수집

mujoco 기반 시뮬레이션 환경에서 `models` 경로에 있는 RB-Y1 모델 파일을 불러와 동작을 시뮬레이션한다. 

```cmd
python record_sim_episodes.py
```

코드에서 default를 바꾸고 실행해도 되고 명령어에서 파라미터 고쳐도 가능  
**예시:**

```cmd
python record_sim_episodes.py --dataset_dir /mnt/storage/jwpark/mobile_aloha/datasets/ --onscreen_render
```

**파라미터:**
- `--task_name`: 데이터 수집할 Task 설정으로 현재는 `sim_transfer_cube`만 구현
- `--dataset_dir`: 데이터셋 저장할 위치 설정
- `--num_episodes`: 반복 에피소드 수
- `--onscreen_render`: 설정시 데이터 수집하면서 시각화

**데이터 구조(.HDF5)**

- `observations/qpos`: 로봇의 관절 위치 (joint positions)
- `observations/qvel`: 로봇의 관절 속도 (joint velocities)
- `observations/images/<cam>`: \<cam> 위치로 촬영한 RGB 이미지
- `action`: 양손 엔드이펙터 위치·자세·그리퍼 제어값으로 구성된 16차원 제어 명령

### 수집된 데이터로 모방학습

```cmd
python imitate_episodes.py
```

**일반 파라미터:**

| 파라미터                   | 타입      | 설명                                          |
| ---------------------- | ------- | ------------------------------------------- |
| `--eval`               | `bool`  | 평가 모드 실행 여부. 지정 시 학습 없이 평가만 수행              |
| `--onscreen_render`    | `bool`  | 평가 중 시각화 렌더링                  |
| `--ckpt_dir`           | `str`   | 학습 체크포인트 저장 경로                              |
| `--policy_class`       | `str`   | 사용할 정책 클래스 이름 (`ACT` 고정)                    |
| `--task_name`          | `str`   | 실행할 시뮬레이션 task 이름 (ex. `sim_transfer_cube`) |
| `--batch_size`         | `int`   | 학습/검증 배치 크기                                 |
| `--seed`               | `int`   | 학습 재현성을 위한 랜덤 시드                            |
| `--num_steps`          | `int`   | 전체 학습 반복 횟수 (iteration)                     |
| `--lr`                 | `float` | 학습률 (learning rate)                         |
| `--load_pretrain`      | `bool`  | 사전 학습된 정책 가중치를 불러올지 여부                      |
| `--eval_every`         | `int`   | 몇 step마다 policy 평가 수행할지 설정                  |
| `--validate_every`     | `int`   | 몇 step마다 validation loss 계산할지 설정            |
| `--save_every`         | `int`   | 몇 step마다 policy를 저장할지 설정                    |
| `--resume_ckpt_path`   | `str`   | 중단된 학습을 이어하기 위한 체크포인트 경로                    |
| `--skip_mirrored_data` | `bool`  | 좌우 반전된 데이터를 학습에서 제외할지 여부                    |
| `--chunk_size`         | `int`   | 1회에 예측할 액션 시퀀스 길이 (ACT의 query 수)            |

**ACT 파라미터:**

| 파라미터                | 타입     | 설명                                      |
| ------------------- | ------ | --------------------------------------- |
| `--kl_weight`       | `int`  | VAE KL divergence 손실의 가중치               |
| `--hidden_dim`      | `int`  | Transformer 모델 내부 hidden size           |
| `--dim_feedforward` | `int`  | FFN(feedforward network)의 크기            |
| `--temporal_agg`    | `bool` | 시계열 정보를 추가로 집계할지 여부 (미사용 시 무시됨)         |
| `--use_vq`          | `bool` | VQ-VAE 기반 latent action 샘플링 활성화 여부      |
| `--vq_class`        | `int`  | VQ dictionary size (클래스 개수)             |
| `--vq_dim`          | `int`  | 각 VQ embedding의 차원                      |
| `--no_encoder`      | `bool` | encoder를 사용하지 않고 random latent를 사용할지 여부 |
