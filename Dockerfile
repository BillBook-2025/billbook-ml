# 1. CUDA + cuDNN runtime (Python 없음 → Python 설치 가능)
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Python 설치
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리
WORKDIR /app

# 4. requirements 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4-1. GPU PyTorch 설치 (CUDA 12.1)
# RUN pip3 install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
#     --extra-index-url https://download.pytorch.org/whl/cu121

# 5. 소스 복사
COPY . .

# 6. FastAPI 실행 (uvicorn)
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]