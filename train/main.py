"""
[1] Config / Imports / 환경 설정
[2] 데이터 로딩 & 전처리 (Dataset 클래스)
[3] 모델 정의 (Transformer / Embedding / MLP 등)
[4] 학습 유틸 (train_one_epoch, evaluate, pos_weight 계산 등)
[5] train_model() 메인 루프
[6] 실행 스크립트 (Run!!) — CFG 설정하고 train_model 호출
"""

# run_train.py
import torch
from config import CFG, num_feat_columns, SEQ_VOCAB_SIZE, TARGET_VOCAB_SIZE
from model import TabularSeqAttentionModel
from train import train_model
from utils import seed_everything, compute_pos_weight, debug_tensor
import numpy as np

# 1️⃣ Seed 고정
seed_everything(CFG['SEED'])

# 2️⃣ pos_weight 계산
# train_labels = np.array([...])  # 실제 train target 값
# pos_weight = compute_pos_weight(train_labels)

# 3️⃣ 학습 중 디버그
# debug_tensor("seq_items", seq_items)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TabularSeqAttentionModel(
    n_num=len(num_feat_columns),
    cat_cardinalities=None,
    seq_vocab_size=SEQ_VOCAB_SIZE,
    target_vocab_size=TARGET_VOCAB_SIZE
).to(device)

train_model(
    model=model,
    TRAIN_PATH='/home/0uk/toss/toss_data/train.parquet',
    feature_cols=num_feat_columns,
    seq_col='seq',
    target_col='clicked',
    device=device,
    pos_weight=51.42
)