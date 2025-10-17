"""
CFG, device 설정, seed 고정,
PAD_IDX, vocab size, feature 정의 등 전역 설정
"""

# config.py
import json

# -----------------------------
# PAD / UNK 토큰
# -----------------------------
PAD_IDX = 0
UNK_IDX = 1
MAX_SEQ_LEN = 1024

# -----------------------------
# 학습 하이퍼파라미터
# -----------------------------
CFG = {
    'BATCH_SIZE': 1024,
    'EPOCHS': 3,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'ACCUMULATION_STEPS': 4
}

# -----------------------------
# 피처 정의
# -----------------------------
l_feat_columns = [f"l_feat_{i}" for i in range(1,28)]
num_feat_columns = l_feat_columns + \
                   [f"feat_a_{i}" for i in range(1,19)] + \
                   [f"feat_b_{i}" for i in range(1,7)] + \
                   [f"feat_c_{i}" for i in range(1,9)] + \
                   [f"feat_d_{i}" for i in range(1,7)] + \
                   [f"feat_e_{i}" for i in range(1,11)] + \
                   [f"history_a_{i}" for i in range(1,8)] + \
                   [f"history_b_{i}" for i in range(1,31)]

cat_columns = ['gender','age_group','day_of_week','hour','inventory_id']
cat_cardinalities = [2, 8, 7, 24, 96]  # embedding vocab size

# -----------------------------
# vocab size (json 등에서 로드 가능)
# -----------------------------
info = json.load(open("vocab_stats.json"))
SEQ_VOCAB_SIZE = info["seq_vocab_size"]
TARGET_VOCAB_SIZE = SEQ_VOCAB_SIZE