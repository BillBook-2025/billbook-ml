# dataset.py
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
import numpy as np
from config import PAD_IDX, UNK_IDX, MAX_SEQ_LEN

class ClickIterableDataset(IterableDataset):
    def __init__(self, parquet_path, feature_cols, cat_cols, seq_col, target_col,
                 seq_vocab=None, target_vocab=None, batch_size=500_000, val=False, seed=42):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.batch_size = batch_size
        self.val = val
        self.seed = seed
        self.seq_vocab = seq_vocab or {}
        self.target_vocab = target_vocab or {}

    def encode_seq(self, seq_str):
        tokens = seq_str.split(",")
        idxs = [self.seq_vocab.get(t, UNK_IDX) for t in tokens]
        if len(idxs) > MAX_SEQ_LEN:
            idxs = idxs[-MAX_SEQ_LEN:]
        if len(idxs) < MAX_SEQ_LEN:
            idxs = [PAD_IDX]*(MAX_SEQ_LEN - len(idxs)) + idxs
        return idxs

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        rng = np.random.RandomState(self.seed)
        for i, batch in enumerate(pf.iter_batches(batch_size=self.batch_size)):
            is_val_batch = (i % 10 == 0)
            if self.val != is_val_batch:
                continue
            df = batch.to_pandas()
            X_num = df[self.feature_cols].astype(np.float32).fillna(0).values
            X_cat = df[self.cat_cols].astype(str).applymap(lambda x: int(x) if x.isdigit() else 0).values
            y = df[self.target_col].astype(str).map(lambda t: self.target_vocab.get(t, UNK_IDX)).values
            seqs = df[self.seq_col].astype(str).values
            if not self.val:
                idx = rng.permutation(len(df))
                X_num, X_cat, y, seqs = X_num[idx], X_cat[idx], y[idx], seqs[idx]
            for xi, ci, si, yi in zip(X_num, X_cat, seqs, y):
                seq_idx = self.encode_seq(si)
                yield (torch.tensor(xi, dtype=torch.float32),
                       torch.tensor(ci, dtype=torch.long),
                       torch.tensor(seq_idx, dtype=torch.long),
                       torch.tensor(yi, dtype=torch.long))

def collate_fn(batch):
    X_num = torch.stack([b[0] for b in batch])
    X_cat = torch.stack([b[1] for b in batch])
    X_seq = torch.stack([b[2] for b in batch])
    y = torch.stack([b[3] for b in batch])
    attn_mask = (X_seq != PAD_IDX).long()
    return X_num, X_cat, X_seq, attn_mask, y