python -m venv .venv
.venv\Scripts\activate
(.venv) PS C:\my-project\fastapi-ml> 일케 뜨면 됨
pip install fastapi torch uvicorn
pip freeze > requirements.txt


uvicorn app.main:app --reload  일케 해서 서버 실행

fastapi_ml_api/
├── app/
│   ├── __init__.py
│   ├── main.py                # 앱 실행 진입점
│   ├── ml_model.py         # 스프링이랑 연결 테스트용
│   ├── api/                   # API 관련 모음
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   └── predict.py   # 스프링이랑 연결 테스트용
│   │   └── v2/
│   │      ├── __init__.py
│   │      └── endpoints/     # 기능별 엔드포인트 파일들
│   │          ├── recommendation.py   # 개인화 추천 API
│   │          ├── clustering.py        # 클러스터링 API
│   │          └── sentiment.py         # 감성 분석 API
│   ├── ml_models/              # ML 모델별 코드 분리
│   │   ├── __init__.py
│   │   ├── recommendation.py  # 추천 모델 로딩, 예측 함수
│   │   ├── clustering.py      # 클러스터링 모델
│   │   └── sentiment.py       # 감성 분석 모델
│   ├── utils/                 # 유틸 함수 모음
│   └── tests/                 # 테스트 코드
│       ├── __init__.py
│       ├── test_recommendation.py
│       ├── test_clustering.py
│       └── test_sentiment.py
├── requirements.txt
└── README.md