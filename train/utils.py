# # utils.py
# import random
# import os
# import numpy as np
# import torch

# # -----------------------------
# # 1️⃣ Seed 고정
# # -----------------------------
# def seed_everything(seed: int = 42):
#     """
#     재현성을 위해 모든 라이브러리의 시드를 고정합니다.
#     """
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     print(f"[Seed set] seed={seed}")

# # -----------------------------
# # 2️⃣ 불균형 데이터 보정용 pos_weight 계산
# # -----------------------------
# def compute_pos_weight(labels: np.ndarray):
#     """
#     BCEWithLogitsLoss에서 사용할 pos_weight를 계산합니다.
#     pos_weight = #negative / #positive
#     """
#     labels = labels.astype(np.float32)
#     pos = labels.sum()
#     neg = len(labels) - pos
#     pos_weight = neg / pos if pos > 0 else 1.0
#     print(f"[Pos Weight] pos={pos}, neg={neg}, pos_weight={pos_weight:.3f}")
#     return pos_weight

# # -----------------------------
# # 3️⃣ 디버그/로깅 함수
# # -----------------------------
# def debug_tensor(name: str, tensor: torch.Tensor):
#     """
#     텐서의 shape, dtype, min/max 값 출력
#     """
#     print(f"[DEBUG] {name}: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min().item()}, max={tensor.max().item()}")