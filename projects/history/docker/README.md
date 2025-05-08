# RB-Y1 시뮬레이터 도커

기존 레인보우로보틱스에서 제공하는 시뮬레이터에서 모델 파일을 수정하는 방법

### 디렉토리 구조

```
.
├── Dockerfile #도커 파일
├── docker-compose.yml #도커 컴포즈 파일
├── app/
| ├── assets #시뮬레이터 UI 이미지 리소스
│ ├── models #Mujoco 로봇 모델 파일
│ ├── app_main #실행 파일
│ └── libmujovo.do.3.2.0 # 기타 필요한 리소스
```

### 도커, 도커 컴포즈 설치

```cmd
sudo apt update
sudo apt install -y docker docker-compose-v2
```

### 순서

1. `rby1a_mujoco` 경로의 모델 파일 수정
2. 도커 이미지 빌드

    ```cmd
    docker build -t rby1-custom:v1.1 .
    ```

3. `docker-compose.sim.yaml` 도커 컴포즈 파일 수정

    ```docker
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

    ```cmd
    docker compose -f docker-compose.sim.yaml up
    ```

5. 만약 다음 에러 발생시
    ```cmd
    rby1-sim-1  | Authorization required, but no authorization protocol specified
    rby1-sim-1  | ERROR: Could not initialize GLFW
    rby1-sim-1  | 
    rby1-sim-1  | Press Enter to exit ...
    rby1-sim-1 exited with code 1
    ```
    명령어 입력 후 재실행

    ```cmd
    xhost +
    ```