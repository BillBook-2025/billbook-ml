"""
얜 Transformer에 들어가는 입력이 시퀀스 인덱스를 더해줘서....
토큰의 순서를 모델이 알 수 있게 하는 클래스래
트랜스포머 자체는 순서 정보를 갖고 있지 않다네??

sin, cos 함수를 통해 각 위치마다 고유 벡터를 생성함
아 우리가 위치 정보를 알아서 만들어줘야하구나
"""


"""
일단 지금 모델에선... static을 transformer에 직접적으로 넣지 않고
따로 저장해놨다가 맨 마지막 mlp 쯤에 concat함

⚙️ 하지만 변형도 있음

물론 경우에 따라서는 static을 transformer에 “간접적으로” 넣기도 합니다.

예시 1: Static을 CLS 토큰으로 삽입

👉 static을 [CLS] 같은 특별 토큰으로 시퀀스 맨 앞에 붙임

cls_token = self.static_proj(static).unsqueeze(1)  # (B, 1, emb_dim)
seq_emb = torch.cat([cls_token, seq_emb], dim=1)
seq_out = self.transformer(seq_emb, attn_mask=mask_with_cls)


이렇게 하면 transformer가 static 정보도 attention으로 볼 수 있어요.
(이건 “BERT-style” CTR 모델에서 자주 씀)

예시 2: Static을 Attention Query로 사용

👉 시퀀스는 Key/Value, static은 Query로 써서
정적 특성 기반으로 “sequence context를 요약”할 수도 있어요.
"""



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class TabularSeqAttentionModel(nn.Module):
    """
    [범주형 x_cat] ─▶ [각 Embedding → concat → cat_proj → emb_dim]
    [숫자형 x_num] ─▶ [num_proj → emb_dim]
                       │
                       ▼
              concat(cat_proj, num_proj) ─▶ static_proj → emb_dim
    
    [시퀀스 seq_items] ─▶ [Embedding → PosEnc → Transformer → pooled]
    [타겟 target_item] ─▶ [Embedding → emb_dim]
    
    static(emb_dim) + target(emb_dim) + pooled(emb_dim)
        │
        ▼
      MLP → logit
    """
    # 실제 계산은 concat 등등... forward에서!! init에선 걍 구조 정의랑 변수 크기 정도..?
    def __init__(
        self,
        n_num,               # number of continuous numeric features -> numeric input layer 필요
        cat_cols,
        seq_vocab,           # vocab for sequence items (include +1 for padding index 0) -> seq emb layer 생성용
        cat_vocabs,
        emb_dim=64,          # base embedding dim -> 모든 emb 차원 통일 기준값
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff=256,
        seq_max_len=100,
        mlp_hidden=[512,256,128],
        dropout=0.2
    ):
        super().__init__()
        
        # --- embeddings for static categorical features ---
        self.cat_embs = nn.ModuleList([ # 각 카테고리마다 따로 임베딩 해야함
            nn.Embedding(
                num_embeddings=len(cat_vocabs[col]),
                # embedding_dim = min(emb_dim, max(4, (len(cat_vocabs[col])+1) // 10))
                # 카테고리 개수에 따라 임베딩 차원 설정하는건데 이거 보다 아래께 더 좋대
                # min.. 이건 단순히 10으로 나누는거라 작은 vocab에선 너무 작고 큰 vocab에선 부족할수도 있음
                embedding_dim=int(min(600, round(1.6 * len(cat_vocabs[col]) ** 0.56)))
                # 얜 ^0.56덕에 작은 vocab엔 작게,, 큰거엔 적절히 크게 비선형적으로 나온다는데?
            )
            for col in cat_cols
        ]) # 결과: (B, emb_i_dim)
        """
        일단 지금은 static 카테고리 임베딩 크기가 제각각임 근데 숫자형 피처는 바로 nn.Linear(n_num, emb_dim) 일케 프로젝션함!
        우리 트랜스포머는 seq emb던지 tgt emb 던지 모두 emb_dim를 기준으로 하는데 지금 상황으론 static 임베딩이랑 emb_dim랑 다름
        + concat된 static emb 크기가 제각각이라 피쳐마다 중요도도 달라지고
        ++ 모델이 학습하면서 정적  feat랑 seq feat랑 중요도 구문이 불명확해짐
        ===> 아! 그럼 cat emb를 emd_dim으로 proj하면 되구나!
        """
        self.cat_proj = nn.Linear(
            sum(e.embedding_dim for e in self.cat_embs), # 각 cat embs를 concat하면 이정도 값인가봐
            emb_dim # 얘로 축소
        )

        # projection for numeric features -> project to some latent dim
        # self.n_num = n_num
        if n_num > 0:
            self.num_proj = nn.Sequential(
                # nn.BatchNorm1d(n_num),
                # nn.LayerNorm(n_num), # 마지막 차원 기준으로 정규화
                nn.Linear(n_num, emb_dim),
                nn.ReLU() # 얘는 왜 렐루 해주는겨?
                # -> 숫자형 입력은 scale이 엄청 차이날 수도 있어서
                # 미리 ReLU해서 좀 값 조정하는건가봐 (경험적인건가봄!)
            ) # 결과: (B, emb_dim)
        else:
            self.num_proj = None

        # --- sequence & target embeddings ---
        self.seq_item_emb = nn.Embedding(
            len(seq_vocab),
            emb_dim, # seq는 하나의 큰 vocab라서 한 개의 임베딩만 있으면 됨
            padding_idx=0 # 근데 row마다 길이 다를테니 패딩해주자
        )
        self.target_emb = self.seq_item_emb # (B, L, emb_dim)

        # positional encoding 앤 머지 근데
        self.pos_enc = PositionalEncoding(emb_dim, max_len=seq_max_len)

        # transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=transformer_heads, 
            dim_feedforward=transformer_ff, 
            dropout=dropout, 
            batch_first=True # (B, L, D) 형태 유지
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)

        # MLP after concat (mlp 넣기 직전에 concat 하는거임)
        # compute static embedding dim
        # 아 얘를 왜 하나 싶었네 ㄷㄷㄷㄷㄷㄷ
        # 지금 모델은 트랜스포머 안에 static 피쳐를 안넣는다네???~!~
        total_static_dim = (emb_dim if len(cat_cols) > 0 else 0) + (emb_dim if n_num > 0 else 0)
        # ← concat 후 최종 static feature dim (cat_proj + num_proj)
        self.static_proj = nn.Linear(total_static_dim, emb_dim)  # emb_dim으로 축소
        self.static_proj_ln = nn.LayerNorm(emb_dim)
        self.static_proj_dropout = nn.Dropout(dropout)

        # --- final input dim for MLP ---
        final_input_dim = 3*emb_dim # static_dim + tgt_dim + pooled_dim (pooled_history?)
        self.layer_norm = nn.LayerNorm(final_input_dim)  # combined_dim = num + cat + seq feature dim
        
        mlp_layers = []
        inp = final_input_dim
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
            inp = h
        mlp_layers += [nn.Linear(inp, 1)] # CTR 예측 logit
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_num, x_cat, seq_items, pad_mask, target_item, return_attn=False): # 클래스 안 함수라서 맨 텀에  self 넣음
        """
        x_num: (B, n_num) float or None
        x_cat: (B, n_cat) long or None
        seq_items: (B, L) long (padding idx=0)
        attn_mask: (B, L), True where padding
        target_item: (B,) long
        """
        B, L = seq_items.shape
        
        # --- static embeddings ---
        static_parts = []
        if x_cat is not None:
            cat_parts = []
            for i, emb in enumerate(self.cat_embs):
                if i < x_cat.shape[1]:  # x_cat의 범위 안에서만 인덱스를 사용하도록 체크
                    cat_parts.append(emb(x_cat[:, i]))  # 각 범주형 특성에 대해 임베딩을 적용
            cat_embs = torch.cat(cat_parts, dim=1)
            static_parts.append(self.cat_proj(cat_embs))
        if x_num is not None and self.num_proj is not None:
            static_parts.append(self.num_proj(x_num))  # (B, emb_dim)
            
        if len(static_parts) > 0:
            static = torch.cat(static_parts, dim=1)
            static = F.layer_norm(static, static.shape[1:])
        else:
            static = torch.zeros(B, 0, device=seq_items.device)
        
        # --- sequence encoding with transformer ---
        seq_e = self.seq_item_emb(seq_items)  # (B, L, emb_dim) B,L을 넣었으니 B,L,emb_dim이 나오지
        seq_e = self.pos_enc(seq_e)           # add positional info 순서 정보 부여

        # Transformer output: (B, L, emb_dim)
        seq_out = self.transformer(seq_e, src_key_padding_mask=pad_mask)

        # --- attention pooling using target embedding as query ---
        tgt = self.target_emb(target_item)  # (B, emb_dim)

        """
        Pooling이란? 여러개의 벡터를 하나라 합치는 연산!
        ex: 시퀀스 임베딩 (B,L,D) -> 평균or합 (B, D)
        하지만 평균이나 합은 모든 시퀀스를 똑같이 봐서 중요한 토큰과 중요하지 않은 토큰을 구분하지 못함
        
        Attention Pooling은? 시퀀스의 각 벡터(seq_out)에 가중치를 주고 합쳐서 하나로 만듦
        이 가중치는... 어떤 토큰이 중요한지를 모델이 학습해서 결정함!
        """
        pooled, attn_w = self.attention_pooling(seq_out, tgt, pad_mask)
        
        # --- combine: static + target_emb + pooled_history ---
        # 최종 mlp 전에 concat
        static_fused = self.static_proj(static)
        static_fused = self.static_proj_ln(static_fused)
        static = self.static_proj_dropout(static_fused)
        combined = torch.cat([static, tgt, pooled], dim=1)
        # combined = self.layer_norm(combined)  # LayerNorm 적용
        logit = self.mlp(combined).squeeze(1)
        
        # return logit
        if return_attn:
            return logit, attn_w
        else:
            return logit


# 🔤 기본적인 텐서 표기 규칙
# 기호	의미	예시 (직관적 비유)
# B	Batch size (배치 크기)	한 번에 학습에 들어가는 “데이터 샘플 개수”
# L	Sequence Length (시퀀스 길이)	한 사용자의 “행동 기록 길이” (예: 본 상품 20개 → L=20)
# D	Embedding Dimension (임베딩 차원)	각 item/feature를 나타내는 벡터 크기 (예: 64차원 벡터)
# n_num	숫자형 피처 개수	예: price, age, rating_count 등 3개면 n_num=3
# n_cat	범주형 피처 개수	예: gender, region, device_type 등 5개면 n_cat=5
# 📘 예시로 보는 입력 형태
# 변수	Shape	설명
# x_num	(B, n_num)	숫자형 피처들 (예: [가격, 나이, 평점])
# x_cat	(B, n_cat)	범주형 피처들 (예: [성별, 지역, 기기])
# seq_items	(B, L)	사용자별 시퀀스 (예: [아이템1, 아이템2, ..., 패딩])
# attn_mask	(B, L)	시퀀스 중 padding 위치를 True로 표시
# target_item	(B,)	이번에 예측할 “현재 타겟 아이템”
# 📈 예시로 감각적으로 이해하기

# 예를 들어, 한 번의 학습 배치(batch)가 3명 사용자 데이터로 구성돼 있다고 하자:

# 사용자	seq_items	target_item
# user1	[15, 92, 33, 0, 0]	12
# user2	[45, 8, 0, 0, 0]	23
# user3	[67, 14, 88, 22, 19]	88

# B = 3 (3명 사용자)

# L = 5 (sequence 길이 5 — padding 포함)

# 각 seq_item은 vocab index (정수)

# 0은 padding index
# → mask로 표시해야 함 (True = padding 자리)

# 🧠 모델 내부에서 shape 흐름

# Transformer 들어가기 전후로 텐서 shape이 계속 변함:

# 단계	텐서명	shape	설명
# 입력	seq_items	(B, L)	아이템 인덱스들
# 임베딩	seq_item_emb(seq_items)	(B, L, D)	각 아이템을 D차원 벡터로 변환
# PosEnc 추가	seq_e + pe	(B, L, D)	위치 정보 추가
# Transformer 출력	seq_out	(B, L, D)	시퀀스 문맥 반영된 벡터
# Attention pooling	pooled	(B, D)	타겟 기준으로 요약된 시퀀스
# Static 피처	static	(B, D)	숫자형 + 범주형 통합 피처
# Target embedding	tgt	(B, D)	예측 타겟의 임베딩
# MLP 입력	[static, tgt, pooled] concat	(B, 3*D)	모든 정보를 합침
# 출력	logit	(B,)	각 샘플의 클릭 확률 로짓
    def attention_pooling(self, seq_out, tgt, pad_mask):
        """
        seq_out: (B, L, D) Transformer 출력
        tgt: (B, D) target embedding
        pad_mask: (B, L) True = padding
        """
        # ==== attention 공식 기억나지? Q·K^T / √d → softmax → * V 그건가봐 ====
        # ======== 거기의 Q -> tgt, (K, V) -> seq_out이 맡은 거임 ========
        # seq_out: (B, L, D), tgt: (B, D)
        # tgt.unsqueeze: (B, 1, D) -> 걍 seq_out과 차원 맞게 하려구
        scores = (seq_out * tgt.unsqueeze(1)).sum(-1) # 각 시퀀스 토큰과 target의 유사도 점수 (B, L)
        # (seq_out * tgt.unsqueeze(1)).shape -> B, L, D!!! .sum(-1)하면 마지막 차원 D를 모두 더함! (B, L)
        d = seq_out.size(-1) # 이거 그 루트 d 기억남? 차원 줄여서 gradient 안정화 하는거
        scores = scores / math.sqrt(d) # (B, L) scaled dot-product, Transformer attention
        scores = scores.masked_fill(pad_mask, float("-inf")) # padding 부분은 -inf를 써서 softmax에서 가중치가 0 나오게

        '''
        # Label-aware attention
        # 이게 지금 클릭 비율이 낮으니까.... attention 레벨에서도 클릭에 대한 가중치를 부여해서
        # 클릭이 실제로 되었으면 그 샘플의 토큰을 가중하는건데....
        # 일단 얘도 나중에 고려...
        if self.training and label is not None:
            pos_mask = (label == 1).float().unsqueeze(1)  # (B, 1)
            scale = 1 + 0.5 * pos_mask  # label==1이면 scores 1.5배 증가
            scores = scores * scale
        '''
        
        attn_w = torch.softmax(scores, dim=1) # (B, L) -> 각 시퀀스 토큰에 대한 가중치
        pooled = torch.einsum("bl, bld -> bd", attn_w, seq_out) # (B, D) 최종적으로 V를 곱하고 sum하는 과정임
        # attn_w를 각 시퀀스 토큰(seq_out; V)에 곱해서 가중 편균을 구함 -> target 기준으로 요약된 시퀀스 표현
        # 결국 “이 사용자의 과거 행동 중 어떤 게 현재 target item과 관련이 높은가?”를 의미
        # ==============================================================
        return pooled, attn_w