# RB-Y1 시뮬레이터 도커

- 기존 레인보우로보틱스에서 제공하는 시뮬레이터에서 모델 파일을 수정하는 방법  
- 도커 내부에서 확인해보면 `app_main` 파일 형태로 실행 파일로만 제공하기 때문에 수정에는 한계가 있음
- RB-Y1 파이썬 SDK로 관절의 **위치**, **속도**, **전류**, **토크값** 확인 가능
- 하지만 카메라를 추가하고 이미지를 받아오는 것은 실행 파일이 제공하지 않는한 어려울 것으로 보여 ALOHA 학습 모델을 적용하려면 이미지를 뺀 방식으로 고려하거나 새로운 시뮬레이터를 구현해야할 것 같아 보임

### 디렉토리 구조

```bash
.
├── Dockerfile #도커 파일
├── docker-compose.yml #도커 컴포즈 파일
├── app/
| ├── assets #시뮬레이터 UI 이미지 리소스
│ ├── models #Mujoco 로봇 모델 파일
│ ├── app_main #실행 파일
│ └── libmujoco.so.3.2.0 #Mujoco 시뮬레이션 엔진 라이브러리
```

### 도커, 도커 컴포즈 설치

```bash
sudo apt update
sudo apt install -y docker docker-compose-v2
```

### 순서

1. `rby1a_mujoco` 경로의 모델 파일 수정
2. 도커 이미지 빌드

    ```bash
    docker build -t rby1-custom:v1.1 .
    ```

3. `docker-compose.sim.yaml` 도커 컴포즈 파일 수정

    ```yaml
    version: '3.8'
    services:
      rby1-sim:
        image: rby1-custom:v1.1 #새로 빌드한 이미지로 수정
        environment:
          - DISPLAY=${DISPLAY}
        volumes:
          - /tmp/.X11-unix:/tmp/.X11-unix
        network_mode: host
        # ports:
        #   - "50051:50051"
    ```

4. 도커 컴포즈 실행

    ```bash
    docker compose -f docker-compose.sim.yaml up
    ```

5. 만약 다음 에러 발생시
    ```bash
    rby1-sim-1  | Authorization required, but no authorization protocol specified
    rby1-sim-1  | ERROR: Could not initialize GLFW
    rby1-sim-1  | 
    rby1-sim-1  | Press Enter to exit ...
    rby1-sim-1 exited with code 1
    ```
    명령어 입력 후 재실행

    ```bash
    xhost +
    ```