# """ Imports """
# import os
# import random
# import math
# import json
# import pickle
# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
# from tqdm.notebook import tqdm
# from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import IterableDataset, DataLoader, SubsetRandomSampler
# from torch.nn.utils.rnn import pad_sequence
# from torch.amp import autocast, GradScaler

# # 시퀀스/타겟 vocab 관련
# # 걍 vocab_size랑 seq_vocab_size랑 뭐가 다른거지?
# with open(PATHS['CAT_VOCAB'], "rb") as f:
#     cat_vocabs = pickle.load(f)

# with open(PATHS['SEQ_VOCAB'], "rb") as f:
#     seq_vocab_dict = pickle.load(f)
# # 아아아아 ㅇㅋ 얘는 seq에 든 inventory_id 다 뒤져보면서... 자주 나오는 애들만 모아두고 그 갯수 세어놓은거임
# # collate_fn()로 데이터셋 처리할때 seq가 너무 크면 모델 터지니까 잘 안등장하는 seq에 대해선 UNK 토큰으로 처리하는데
# # 그떄 vacab_size 값이 쓰임!!
# # 그리고 또... 우리가 다루는 데이터의 개수가 vocab_size이니까 최종 출력때 나올 inventory_id에도 관여함

# info = json.load(open(PATHS['VOCAB_STATS']))
# total_vocab_size = info["seq_max"] + 2
# # 얘는 임베딩 전요 변수임! 전체 seq 뒤져보고 전체 개수 구해서 임베딩 차원 정할때 씀!!
# # 근데 어차피 학습을.... vocab_size로 할껀데... 임베딩을 얘를 기준으로 해도 되는건가?????
# # 생각해보면 출력 레이어도 필터링된 거를 쓰니까.... 딱히 쓸모는 없음
# """
# 찾아보니 실제 ctr 서비스에선 hash bucket이라는 개념을 써서....
# 자주 안 등장하는 애들을 전부 UNK 토큰으로 처리하는게 아니라
# 여러 hash bucket에 나누어 담아가지고 학습을 처리한다네?
# 그래서 어~~~~엄청 희귀한 애들안 UNK으로 처리하고 나머진 해시버킷에 넣어두 되고.... 머 그렇대

# 아 그리고 또.. 아까 hash bucket으로 돌아가서....
# 얘를 걍 수학적인 랜덤이 아니라 뭔가 inventory_id를 클러스터링 같은거 해서 처리할 순 없을까?
# 뭐 할 순 있는데... 이런건 성능 비교해보고 하래..!!!!! DeepHash? 이런것도 한번 고려해보고
# """

# VOCABS = {
#     'SEQUENCE': seq_vocab_dict,
#     'CATEGORICAL': cat_vocabs
# }

# # -----------------------------
# # 모델 선언 아 예전 코드는 이렇게 전역으로 선언했었구나.....
# # -----------------------------
# model = TabularSeqAttentionModel( 
#     n_num=len(FEATURE_COLS['NUMERIC']),   # 숫자형 feature 수 
#     cat_cols=FEATURE_COLS['CATEGORICAL'], 
#     seq_vocab=VOCABS['SEQUENCE'],         # sequence token embedding 
#     cat_vocabs=VOCABS['CATEGORICAL'], 
#     emb_dim=64,            # embedding dim 이거 seq+vocab용 임베딩 차원임 나중에 공부 ㄱㄱ 
#     transformer_layers=2, 
#     transformer_heads=4, 
#     transformer_ff=256, 
#     seq_max_len=1024,      # truncation/pad 기준 
#     mlp_hidden=[512,256,128], 
#     dropout=0.2 
# ) # .to(device)

# """ Run!! """
# # GPU assert 위치 잡기
# import os
# # os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_model(PATHS, FEATURE_COLS, VOCABS, CFG, device=device)