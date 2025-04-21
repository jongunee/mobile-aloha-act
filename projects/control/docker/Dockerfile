# 빌드 시 사용할 ARG 변수 (필요 시 값을 전달)
ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH

FROM ubuntu:22.04

# 이미지 메타데이터 설정
LABEL org.opencontainers.image.ref.name="ubuntu"
LABEL org.opencontainers.image.version="22.04"

# (옵션) 한국 미러 서버로 변경
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://kr.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://kr.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list

# apt 리스트 캐시 삭제 후 업데이트 실행 및 필요한 라이브러리 설치 (libglfw3, libfreetype6)
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y libglfw3 libfreetype6 && \
    apt-get clean

# 현재 디렉토리의 app 폴더를 /root/exe/app 으로 복사 (app 디렉토리가 존재해야 합니다)
COPY ./app /root/exe/app

# 실행 파일(app_main)에 실행 권한 부여 (실제 파일명이 app_main인지 확인)
RUN chmod +x /root/exe/app/app_main

# 작업 디렉토리 설정
WORKDIR /root/exe/app

# LD_LIBRARY_PATH 환경변수 설정 (시스템 라이브러리 경로 포함)
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/root/exe/app/:

# 컨테이너 시작 시 "./app_main" 실행
CMD ["/bin/bash", "-c", "./app_main"]
