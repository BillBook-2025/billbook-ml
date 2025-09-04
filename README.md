# billbook-ml

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

# 5. 가상환경 활성화(나가는건 deactivate)
source .venv/bin/activate

# 6. pip 업그레이드
python3 -m pip install --upgrade pip

# 7. 필요한 패키지 설치
python3 -m pip install fastapi torch uvicorn
...

# 8. 패키지 버전 저장
pip freeze > requirements.txt

# 9. (다른 환경에서) 패키지 일괄 설치 (venv 활성화 상태)
pip install -r requirements.txt

# 10. VSCode 실행
code . (ctrl+shift+p 눌러서 Python: Select Interpreter 입력 후 venv 선택)

# 11. Jupyter Notebook 실행
jupyter notebook
