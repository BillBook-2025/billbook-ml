# billbook-ml
https://chatgpt.com/share/6914a234-fc4c-800c-9744-ddf3ba4d5f21

## 프로젝트 개요
- 아 머라하지

### 폴더 구조
```
fastapi_ml_api/
├── fastapi_ml_api
│   ├── app.py                  # FastAPI 엔드포인트
│   ├── ml_models/              # ML 모델별 코드 분리
│   │   ├── recommendation.pkl  # 추천 모델 로딩, 예측 함수
│   │   ├── clustering.pkl      # 클러스터링 모델
│   │   └── sentiment.pkl       # 감성 분석 모델
│   └── utils/                  # 유틸 함수 모음
│
├── ml_pipeline
│   ├── train.py                # 학습코드
│   ├── preporocess.py          # 전처리 코드
│   └── export.py               # DB에서 데이터 가져오기
│
├── requirements.txt
└── README.md
```

### 환경설정 명령어 모음 (WSL + Python 가상환경)
```bash
# 1. WSL 실행(종료는 exit)
wsl

# 2. 프로젝트 폴더 생성 및 이동
mkdir myproject && cd myproject

# 3. 가상환경(venv) 설치
sudo apt update
sudo apt install python3.10-venv

# 4. 가상환경 생성
python3 -m venv .venv

# 5. 가상환경 활성화
source .venv/bin/activate
deactivate

# 6. pip 업그레이드 및 필요한 패키지 설치
python3 -m pip install --upgrade pip
python3 -m pip install fastapi torch uvicorn
...

# 7. 패키지 버전 저장
pip freeze > requirements.txt

# 기타..
pip install -r requirements.txt

docker pull goljeol/billbook-fastapi:latest
docker run -d -p 8000:8000 goljeol/billbook-fastapi:latest

code . (ctrl+shift+p 눌러서 Python: Select Interpreter 입력 후 venv 선택)
jupyter notebook
```

### 도커 사용법
```bash
# 8. 도커 빌드
docker build -t billbook-fastapi .

# 9. 도커 허브에 push
docker login
docker tag billbook-fastapi goljeol/billbook-fastapi:latest
docker push goljeol/billbook-fastapi:latest

docker run -d -p 8000:8000 billbook-fastapi → 컨테이너 실행


# docker run -d -p 8000:8000 billbook-fastapi
# docker run -d -p 8001:8000 billbook-fastapi
# docker run -d -p 8002:8000 billbook-fastapi
# 일케 같은 이미지를 여러 도커에 킬 수도 있음
docker logs -f 419b69c7a7dd

docker ps → 실행 확인
docker stop <컨테이너ID> → 컨테이너 종료

# docker run -it --rm -p 8000:8000 billbook-fastapi /bin/bash
# uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# 이건 도커 빌드하다가 생긴 캐쉬들 삭제하는...
# docker image prune -a
# docker builder prune
```