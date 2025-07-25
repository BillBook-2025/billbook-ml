fastapi_project/
│
├── app/
│   ├── __init__.py             # 해당 폴더(app)을 패키지로 인식하게 함
│   │                           # 이거 있어야 이 폴더를 import 할 수 있게 됨
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/                 # 버전별 API 라우터 분리 (v1, v2 ...)
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/      # 실제 엔드포인트 모음
│   │   │   │   ├── users.py
│   │   │   │   ├── items.py
│   │   │   │   └── ...
│   │   │   └── dependencies.py # 공통 의존성 주입 코드
│   │   └── deps.py             # 공통 의존성 (DB 연결 등)
│   │
│   ├── core/                   # 설정, 보안, 시크릿, JWT 등 핵심 기능
│   │   ├── config.py
│   │   ├── security.py
│   │   └── ...
│   │
│   ├── models/                 # DB 모델 (예: SQLAlchemy 모델)
│   │   ├── __init__.py
│   │   └── user.py
│   │
│   ├── schemas/                # Pydantic 스키마 (Request/Response DTO)
│   │   ├── __init__.py
│   │   └── user.py
│   │
│   ├── crud/                   # DB 접근 로직 함수 모음
│   │   ├── __init__.py
│   │   └── user.py
│   │
│   ├── db/
│   │   ├── base.py             # 베이스 모델, DB 초기화
│   │   ├── session.py          # DB 세션 생성 (SQLAlchemy)
│   │   └── ...
│   │
│   ├── utils/                  # 유틸리티 함수 모음
│   └── tests/                  # 테스트 코드 (pytest 등)
│
├── alembic/                    # DB 마이그레이션 (선택)
│
├── main.py                     # FastAPI 앱 엔트리 포인트
├── requirements.txt            # 필요한 패키지 목록
├── README.md
├── .env                        # 환경 변수
└── docker-compose.yml          # 도커 설정 (필요 시)