# """ Config """
# # 학습 관련 하이퍼파라미터
# # 배치 크기, 학습률, 에폭 수, 시드, gradient accumulation 단계 수 등
# CFG = {
#     'BATCH_SIZE': 512, # 1024
#     'EPOCHS': 10,
#     'LEARNING_RATE': 1e-3, # 3e-4
#     'SEED': 42,
#     'ACCUMULATION_STEPS': 4,
#     'CHUNK_SIZE': 300_000, # 500_000
#     'TOTAL_SAMPLES': 10_704_179, # 얘랑
#     'SAMPLE_RATIO': 0.2 # 얘는 걍 tqdm 계산용..
# }

# # 여러 경로들 'key': value,
# PATHS = {
#     'TRAIN_PATH': '/home/0uk/toss/toss_data/train.parquet',
#     'CAT_VOCAB': '/home/0uk/toss/cat_vocabs.pkl',
#     'SEQ_VOCAB': 'inventory_id_vocab.pkl',
#     'VOCAB_STATS': 'vocab_stats.json',
#     'SAVE': '/home/0uk/toss/save',
#     'BEST_MODEL': f"/home/0uk/toss/save/best_{CFG['SAMPLE_RATIO']}_bce.pt"
# }

# # 피처 및 컬럼 정의
# """
# seq 얜 유저 서버 로그(유저가 뭐 한지)
# l_feat 얜 광고의 속성과 관련된 피쳐!(장르, 회사 등) -> 범주화 가능한거!
# - 지금 데이터야 뭐 l_feat도 숫자라 상관없긴 한데.. 막 판타지, 로맨스 같은 장르 같은거면 임베딩 따로 해야하니 구분만..
# - 나중에 feature importance나 ablation 실험할 때도 l_feat만 따로 떼서 테스트 가능
# feat 이런 애들은 정보 영역임!!! 뭐... 어디에 광고 떴는지,, 요약 정보라던지..
# history 얜 과거 인기도 피쳐(인기 순위, 최근 클릭 수 등등)
# """
# FEATURE_COLS = {
#     'CATEGORICAL': ['gender','age_group','day_of_week','hour','inventory_id'],
#     'SEQUENCE': 'seq',
#     'L_FEAT_COL': [f"l_feat_{i}" for i in range(1,28)],
#     'NUMERIC': [f"l_feat_{i}" for i in range(1,28)] + \
#                [f"feat_a_{i}" for i in range(1,19)] + \
#                [f"feat_b_{i}" for i in range(1,7)] + \
#                [f"feat_c_{i}" for i in range(1,9)] + \
#                [f"feat_d_{i}" for i in range(1,7)] + \
#                [f"feat_e_{i}" for i in range(1,11)] + \
#                [f"history_a_{i}" for i in range(1,8)] + \
#                [f"history_b_{i}" for i in range(1,31)],
#     'TARGET': 'clicked',
# }

# """ Constants """
# # 토큰 및 시퀀스 설정
# PAD_TOKEN = 0
# UNK_TOKEN = 1
# MAX_LEN = 1024  # 시퀀스 최대 길이

# # 재현을 위한 시드 설정
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# seed_everything(CFG['SEED'])