# https://chatgpt.com/share/68eabeea-4804-800c-aba4-3dd61453a0a7
"""
해야할거!!!!!!!!!!
1. z-score 표준화 이거... 지금은 청크 단윈데 전체 단위로 바꾸기
2. valid를.... seq 기반으로... 맨 뒤 10퍼를 쓴다거나 해야할듯
   근데 이러면 history_a 같은 피쳐는 어케하지.... ㅋ큐ㅠㅜ
"""

class ClickIterableDataset(IterableDataset):
    def __init__(self, parquet_path, feature_cols, cat_cols, seq_col, target_col,
                 seq_vocab=None, target_vocab=None, cat_vocabs=None, chunk_size=500_000, val=False, seed=42):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.chunk_size = chunk_size
        self.val = val
        self.seed = seed
        self.seq_vocab = seq_vocab or {}        # vocab mappings: inventory_id -> index
        self.target_vocab = target_vocab or {}  # target_item -> index / 0 1 그 레이블 값이 아니라 예측한 iD래
        self.cat_vocabs=cat_vocabs

    def encode_seq(self, seq_str):
        # seq_str이 문자열인 경우, split해서 리스트로 변환
        if isinstance(seq_str, str):
            tokens = seq_str.split(",")
        elif isinstance(seq_str, list):  # seq_str이 리스트인 경우
            tokens = seq_str
        else:
            raise TypeError(f"Expected seq_str to be str or list, but got {type(seq_str)}")
        
        # seq_vocab을 기준으로 인덱스 변환
        idxs = [self.seq_vocab.get(t, UNK_TOKEN) for t in tokens]
        
        # 모든 인덱스가 vocab 범위 내에 있도록 clip
        # max_index = len(self.seq_vocab) - 1
        # idxs = [min(i, max_index) for i in idxs]
        
        # truncation (길이가 길면 최근 MAX_LEN 만큼만 유지)
        if len(idxs) > MAX_LEN:
            idxs = idxs[-MAX_LEN:]
        
        # padding (길이가 짧으면 앞에 PAD_TOKEN으로 채우기)
        if len(idxs) < MAX_LEN:
            idxs = [PAD_TOKEN] * (MAX_LEN - len(idxs)) + idxs
        
        return idxs

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        rng = np.random.RandomState(self.seed)
        
        for i, batch in enumerate(pf.iter_batches(batch_size=self.chunk_size)):

            # # ==========================
            # if i < 20:
            #      continue
            # # ==========================
            
            is_val_batch = (i % 10 == 0)
            if self.val != is_val_batch:
                continue

            df = batch.to_pandas()
            # 숫자형 피처
            X_num = df[self.feature_cols].astype(np.float32).fillna(0).values
            # 문자열 'nan'을 실제 np.nan으로 대체하여 astype이 정상적으로 작동하도록 함
            # X_num_df = X_num_df.replace('nan', np.nan, regex=True)
            # X_num = X_num_df.astype(np.float32).values
            # X_num[np.isinf(X_num)] = 0 # NaN과 Inf를 모두 0으로 대체합니다.
            # X_num[np.isnan(X_num)] = 0
            # # 값의 범위를 제한하여 이상치로 인한 문제를 방지합니다.
            # X_num = np.clip(X_num, -1e6, 1e6) # 예를 들어, -1백만 ~ 1백만 사이로 제한
            # 카테고리형 피처 (gender, age_group, ..., inventory_id)
            # inventory_id 컬럼은 이미 cat_cols에 포함되어 있으므로, X_cat을 그대로 사용

            mask = np.random.rand(len(df)) < CFG['SAMPLE_RATIO'] # 샘플링
            df = df[mask]
            X_num = X_num[mask]

            # Z-점수 표준화
            # 이거 안하니까 autocast 적용하면 BinaryCrossEntropyWithLogitsBackward0' returned nan values... 오류 뜨더라
            # 이유는... num_feat 얘네 정규화 안해서 스케일 커서 그렇대
            mean = X_num.mean(axis=0)
            std = X_num.std(axis=0)
            # 0으로 나누는 오류 방지
            std[std == 0] = 1e-6 
            X_num = (X_num - mean) / std
            
            # X_cat = df[self.cat_cols].astype(str).apply(lambda col: col.map(lambda x: int(x) if x.isdigit() else 0)).values
            # 여기서 각 범주형 컬럼의 값을 그냥 int(x)로 바꾸니까, 예를 들어 91이란 숫자 자체가 그대로 들어가요.
            # 그런데 모델은 이 카테고리 컬럼을 위해 nn.Embedding(2, emb_dim) 같은 아주 작은 크기의 임베딩을 가지고 있음
            # 그래서 91을 인덱스로 쓰려다 GPU 폭발
            # X_cat = np.stack([
            #     df[col].map(lambda x: cat_vocabs[col].get(str(x), 0)).fillna(0).astype(int).values
            #     for col in self.cat_cols
            # ], axis=1)
            X_cat = df[self.cat_cols].apply( # 일케 하는게 더 빠르대 ㄷㄷ 이유는 불명
                lambda col: col.astype(str).map(lambda x: self.cat_vocabs[col.name].get(x, 0))
            ).fillna(0).astype(int).values

            # 클릭 여부 (y: clicked)와 클릭 대상 아이템 ID (target_items: inventory_id) 분리
            y = df[self.target_col].astype(int).values
            target_items = df['inventory_id'].astype(str).apply(lambda x: int(x) if x.isdigit() else 0).values

            # sequence
            # seqs = df[self.seq_col].astype(str).values
            """
            위 코드로 하면 걍 모든 값을 문자열로 변환해서.... NaN이나 빈칸 같은 특수값도 nan이런게 돼서...
            int(seq)나 split(',')에서 예상치 못한 값이 들어가 CUDA 인덱스 에러가 발생했는데...

            아래 코드는 타입과 빈 값을 좀 빡세게 체크해서 숫자로 변환 못하는 값은 다 빼버려서
            CUDA assertion 오류가 안나게 된건가?
            """
            seqs = []
            for seq_raw in df[self.seq_col].values:
                if isinstance(seq_raw, str) and seq_raw.strip():  # 문자열이면서 빈 문자열이 아닌 경우
                    # 쉼표로 구분된 문자열 → int 리스트로 변환
                    seq_list = [int(x) for x in seq_raw.split(',') if x.strip().isdigit()]
                elif isinstance(seq_raw, (list, np.ndarray)):
                    # 리스트나 배열이면 int로 변환 (문자열 숫자도 안전하게)
                    seq_list = [int(x) for x in seq_raw if str(x).isdigit()]
                else:
                    # None, NaN, 빈 값 등 예외 처리
                    seq_list = []
                seqs.append(seq_list)
            
            # # epoch-level class count monitoring
            # pos_count = (y==1).sum()
            # neg_count = (y==0).sum()
            # print(f"Chunk {i} | Pos: {pos_count} | Neg: {neg_count}")

            for xi, ci, si, yi, ti in zip(X_num, X_cat, seqs, y, target_items):
                seq_idx = self.encode_seq(si)

                yield xi, ci, seq_idx, yi, ti # yield일땐 numpy만 반환하는게 좋대
                """
                원래 dataset loop에 있던 torch.tensor 호출을 collator. 즉, dataloader로 옮기면서
                전체 batch 단위로 .to(device)가 되어서 I/O 병복 줄어듦
                """
                # yield (torch.tensor(xi, dtype=torch.float32),
                #        torch.tensor(ci, dtype=torch.long),
                #        torch.tensor(seq_idx, dtype=torch.long),
                #        torch.tensor(yi, dtype=torch.long), # 클릭 여부
                #        torch.tensor(ti, dtype=torch.long)) # 클릭 대상 아이템 I

"""
collate_fn은 PyTorch DataLoader가 배치(batch)를 만드는 방법을 정의하는 함수임
기본적으로 DataLoader는 Dataset에서 __getitem__으로 하나씩 데이터를 가져와서 배치 단위로 묶는 역할을 하는데,
숫자/카테고리/시퀀스 등 여러 타입이 섞여 있거나, 시퀀스 길이가 달라 패딩이 필요하거나,
특별한 마스크나 처리(UNK 처리 등)가 필요한 경우
그냥 기본 default_collate로는 처리 못할 때 사용자가 직접 묶는 방법을 정의해야 함
대충 "배치를 만들면서 전처리까지 처리하는 함수"라고 생각하면 되는듯

- 리스트 형태로 들어오는 batch에서 각 텐서를 쌓아(batch_size, ...) 형태로 만듦 (torch.stack)
- seq_items나 target_items를 vocab 범위에 맞게 안전하게 UNK 처리
- 시퀀스 길이 다를 수 있으므로 패딩 마스크 생성 (pad_mask)
- 최종적으로 모델 forward에 맞는 형태로 모든 데이터를 반환
"""
class Collator: # 원래 걍 def collate_fn(batch) 였는데 seq_vocab_size도 넘겨줄겸 class로 정의함
    def __init__(self, seq_vocab_size, UNK_TOKEN=1, PAD_TOKEN=0):
        self.seq_vocab_size = seq_vocab_size
        self.UNK_TOKEN = UNK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

    def __call__(self, batch):
        # X_num = torch.stack([b[0] for b in batch])
        # X_cat = torch.stack([b[1] for b in batch])
        # X_seq = torch.stack([b[2] for b in batch])
        # y = torch.stack([b[3] for b in batch]).long()
        # target_items = torch.stack([b[4] for b in batch]).long()
        # numpy 리스트 → 바로 텐서 변환 + stack (벡터화)
        X_num = torch.from_numpy(np.stack([b[0] for b in batch])).float()
        X_cat = torch.from_numpy(np.stack([b[1] for b in batch])).long()
        X_seq = torch.from_numpy(np.stack([b[2] for b in batch])).long()
        y = torch.from_numpy(np.stack([b[3] for b in batch])).long()
        target_items = torch.from_numpy(np.stack([b[4] for b in batch])).long()

        # X_seq = X_seq.long()
        # target_items = target_items.long()
        
        # # 긴급조치: 임베딩 범위 넘침 방지
        # X_seq = X_seq.clamp(max=total_vocab_size - 1)
        # target_items = target_items.clamp(max=target_vocab_size - 1)
        # 걍 clamp 써도 최대값을 제한하긴 함!! 임베딩 범위 넘치는건 막아주는데....
        # 원래 vocab에 없던 아이템이 그냥 마지막 임베딩(total_vocab_size-1)를 덮어 쓰게 되고
        # 음수값이면 걍 그대로 처리돼서 오류 발생함    
        # 음수나 seq_vocab_size 이상인 index를 UNK_TOKEN으로 대체
        X_seq = torch.where(X_seq >= self.seq_vocab_size, self.UNK_TOKEN, X_seq)
        X_seq = torch.where(X_seq < 0, self.UNK_TOKEN, X_seq)
        target_items = torch.where(target_items >= self.seq_vocab_size, self.UNK_TOKEN, target_items)
        target_items = torch.where(target_items < 0, self.UNK_TOKEN, target_items)

        # create src_key_padding_mask for transformer: True for positions that are PAD (should be masked)
        attn_mask = (X_seq != self.PAD_TOKEN)
        # 트랜스포머에선 우리가 무시할 위치... 패딩으로 패위 실제 의미 없는 값인 곳을 표시해서 넘겨줘야함
        # 따라서 True = 패딩된 위치가 되도록 pad_mask를 반대로 넘겨줘야함 (pad_mask를 attn해라!!)
        pad_mask = ~attn_mask
        # Transformer (batch_first=True) accepts src_key_padding_mask with shape (B, L)
        # Transformer will ignore padded positions.
        
        return X_num, X_cat, X_seq, pad_mask, y, target_items