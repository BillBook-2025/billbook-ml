# # 데이터 준비 관련 코드를 따로 함수로 빼서, batch size, augmentation, shuffle 등 관리
# def get_data_loader(parquet_path, feature_cols, cat_cols, seq_col, target_col, 
#                     seq_vocab, cat_vocabs, chunk_size, batch_size, 
#                     val=False, shuffle=True, augmentations=None):
#     ds = ClickIterableDataset(
#         parquet_path=parquet_path,
#         feature_cols=feature_cols,
#         cat_cols=cat_cols,              # categorical
#         seq_col=seq_col,
#         target_col=target_col,
#         seq_vocab=seq_vocab_dict,
#         # target_vocab=seq_vocab_dict,
#         cat_vocabs=cat_vocabs,
#         chunk_size=chunk_size,
#         val=val
#     )

#     # 데이터 augmentation이 있을 경우 처리 (예: 임의의 트랜스폼 추가)
#     # if augmentations:
#     #     ds = augment_data(ds, augmentations)

#     """
#     1. Transformer에서의 self-attention 계산은 입력 시퀀스 길이 L에 대해 O(L²) 메모리를 사용
#        즉, 시퀀스 길이가 길수록 batch size 1개당 메모리 부담이 훨씬 커짐.

#     2. Train: gradient 계산 + optimizer 업데이트 → 추가 메모리 필요
#        하지만 mixed precision(autocast) + gradient accumulation으로 어느 정도 커버 가능
    
#        Val: gradient는 필요 없지만 attention map과 forward activation이 그대로 남아 있음
#        게다가 autocast 때문에 일부 float16/32 변환, GPU fragmentation 등으로 실제 사용량이 
#        train보다 더 불안정하게 늘어날 수 있음 그래서 같은 batch size라도 val에서 OOM이 더 쉽게 발생

#     3. Transformer + 긴 시퀀스 → val batch size를 train보다 줄이는 것이 안전(메모리 터지면)
#     """
    
#     # DataLoader 설정
#     collate_fn = Collator(len(seq_vocab))
#     dl = DataLoader(
#         ds, 
#         batch_size=batch_size // 4 if val else batch_size,
#         num_workers=0, 
#         collate_fn=collate_fn,
#         pin_memory=True,
#         shuffle=shuffle if not val else False
#     )

#     return dl

# def train_epoch(model, train_loader, criterion, optimizer, scaler, 
#                 accumulation_steps, device, train_steps, epoch):
#     model.train()
#     train_loss, train_samples = 0.0, 0
#     optimizer.zero_grad()

#     torch.autograd.set_detect_anomaly(True)  # backward 에러 추적 켜기
    
#     for step, (x_num, x_cat, seq_items, attn_mask, y, target_items) in enumerate(
#         tqdm(train_loader, desc=f"Train Epoch {epoch}", total=train_steps)
#     ):
#         """
#         Positive 샘플이 극히 적으면 loss 계산 중 gather/index_select 연산에서 0개 샘플을 참조하려고
#         시도할 수 있음 이런 경우 GPU assert가 발생하기도 함 -> 라고 하네 ㄷㄷ
        
#         아니면 mask를 만들어서 loss 계산에서 제외 가능하도록 mask 쓰던가

#         Transformer용 입력 시퀀스 길이가 1024인데, 
#         실제 seq_items는 최대값이 1뿐임 → 대부분 패딩 모델 내부에서 index_select 같은 인덱스 연산이 
#         시퀀스 길이 기준으로 수행되는데, padding 처리나 mask 처리 미흡이면 CUDA assert 발생 가능
#         즉, 실제 의미 있는 시퀀스가 없어서 범위 벗어난 인덱스를 참조하려 시도할 수도 잇대
    
#         logits = model(x_num, x_cat, seq_items, attn_mask, target_items)

#         # mask 적용: y >= 0 (ignore_index=-1)
#         mask = y >= 0   # ignore_index=-1이면 이렇게 처리 가능
#         loss = F.binary_cross_entropy_with_logits(
#             logits[mask], y[mask].float()
#         )
        
#         loss = criterion(logits, y.float())
#         """
#         # if y.sum() == 0:
#         #     continue  # 이 배치는 건너뛰기
        
#         # # 디버깅 로그
#         # if step < 614:  # 확인하고 싶은 배치 번호
#         #     continue
#         """
#         x_cat 값이 0 ~ 91인데,
#         model.cat_embs[0] (범주형 임베딩)의 크기가 2개뿐이에요 (num_embeddings=2).
#         즉, 카테고리 인덱스가 임베딩 테이블 범위를 벗어나서 GPU assert가 터진 것
#         ClickIterableDataset에서의 X_cat 부분에서 터졌나봄 ㅜㅜ
        
#         print(f"[Step {step}] seq_items.max()={seq_items.max().item()} | len(seq_vocab_dict)={len(seq_vocab_dict)}")
#         print(f"[Step {step}] target_items.max()={target_items.max().item()} | model.target_emb.num_embeddings={model.target_emb.num_embeddings}")
#         print(f"[Step {step}] x_cat.max()={x_cat.max().item()} | model.cat_embs[0].num_embeddings={model.cat_embs[0].num_embeddings}")
#         """
        
#         # 데이터 디바이스 이동
#         batch = (x_num, x_cat, seq_items, attn_mask, y, target_items)
#         batch = [t.to(device, non_blocking=True) for t in batch] # 이거랑 pinned _memory랑 궁합 좋다는데?
#         x_num, x_cat, seq_items, attn_mask, y, target_items = batch
#         # x_num, x_cat, seq_items, attn_mask, y, target_items = (
#         #     x_num.to(device), x_cat.to(device), seq_items.to(device), 
#         #     attn_mask.to(device), y.to(device), target_items.to(device)
#         # )

#         # model forward 호출 시, attn_mask를 src_key_padding_mask 인자에 전달
#         # 모델의 forward는 target_item의 임베딩을 쓰므로 target_items를 넘겨줌
#         # loss = criterion(logits, target.float()) / accumulation_steps # BCE는 0~1 float값을 원해서 target.float() 해줌
#         # 아나 근데 with autocast(device_type='cuda'): 하니까 자꾸 값이 0이 되거나 그래버리미;; CTR이라 그런가
#         with autocast(device.type): # forword
#             logits = model(x_num, x_cat, seq_items, attn_mask, target_items)
#             """
#             극단적인 imbalance에서는 loss 폭발 방지용으로
#             label smoothing을 살짝 써도 괜찮다는데??? 이건 뭐 나중에 실험해보고..
#             smooth_target = y * 0.9 + 0.05  # 0→0.05, 1→0.95
#             loss = criterion(logits, smooth_target)
#             """
#             loss = criterion(logits, y.float())  # 로스 계산시에는 클릭 여부 y로 계산
#             loss = loss / accumulation_steps # 평균 reduction 이라면 accumulation을 위해 loss를 나누는게 일반적
        
#         scaler.scale(loss).backward()

#         # 파라미터 업데이트 시점
#         if (step + 1) % accumulation_steps == 0:
#             # 2. Gradient clipping (mixed precision에서는 먼저 unscale)
#             scaler.unscale_(optimizer)  # scaler가 적용한 gradient를 원래 스케일로 되돌림
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#         # 손실(loss)은 클릭 여부(y)를 기준으로 계산되므로, 
#         # 배치 크기(size(0))를 가져올 때도 y를 사용하는 것
#         train_loss += (loss.item() * accumulation_steps) * y.size(0)  # loss는 나눠졌으므로 복원
#         train_samples += y.size(0)
#         """
#         criterion 기본 reduction='mean' → batch 내 평균 loss 반환
#         loss = criterion(...) / accumulation_steps → gradient accumulation 때문에 나눔
#         loss 누적할 때는 실제 총 loss 합계를 맞춰야 전체 평균 계산 가능
#         => ys.size만큼 += 해줌
#         """
#     avg_train_loss = train_loss / train_samples
    
#     return avg_train_loss

# def valid_epoch(model, val_loader, criterion, device, epoch):
#     model.eval()
#     val_loss, val_samples = 0.0, 0
#     all_labels, all_preds = [], [] 
    
#     with torch.no_grad():
#         for steps, (x_num, x_cat, seq_items, attn_mask, y, target_items) in enumerate(
#             tqdm(val_loader, desc=f"Val Epoch {epoch}")
#         ):
#             batch = (x_num, x_cat, seq_items, attn_mask, y, target_items)
#             batch = [t.to(device, non_blocking=True) for t in batch] # 이거랑 pinned _memory랑 궁합 좋다는데?
#             x_num, x_cat, seq_items, attn_mask, y, target_items = batch
#             # x_num, x_cat, seq_items, attn_mask, y, target_items = (
#             #     x_num.to(device), x_cat.to(device), seq_items.to(device), 
#             #     attn_mask.to(device), y.to(device), target_items.to(device)
#             # )

#             # with autocast(device_type='cuda'): # forword
#             # with autocast(device_type='cuda'):를 사용하면 모델 연산이 float16으로 수행되는데, float16은 표현 가능한 숫자 범위가 작아서 
#             # BCEWithLogitsLoss 같이 큰 값이나 극단적인 log 계산에서는 쉽게 NaN이 발생합니다.
#             with autocast(device.type): # forword
#                 logits, attn_w = model(x_num, x_cat, seq_items, attn_mask, target_items, return_attn=True)    
#                 logits = logits.clamp(-20, 20) # logits이 넘 크던 작던 해서 NaN이 되버린다는디
#                 loss = criterion(logits, y.float()) # 로스 계산시에는 클릭 여부 y로 계산

#             val_loss += loss.item() * y.size(0)
#             val_samples += y.size(0)

#             # ===========================================================
#             """
#             모델이 얼마나 손실(loss)을 줄였냐를 넘어서,
#             모델이 실제로 클릭/비클릭을 잘 구분하냐를 평가하기 위해
#             Recall,,, F1 같은 지표를 쓰자!!!

#             또!!! CTR에선 클릭 확률이 높은 아이템을 상단에 올리는게 중요하니까
#             모델이 얼마나 클릭과 비클릭을 잘 순위 매기는가를 평가할 수 있는 AUC도 좋음
#             """
#             # 확률 예측 (sigmoid)
#             probs = torch.sigmoid(logits).detach().cpu().numpy()
#             all_labels.append(y.cpu().numpy())
#             all_preds.append(probs) # 확률값 저장 (AUC용)





            
#             # --- attention 시각화 (옵션) ---
#             if steps == 0:  # 첫 배치만
#                 attn_w_np = attn_w.detach().cpu().numpy()  # (B, L)
#                 import matplotlib.pyplot as plt
#                 plt.figure(figsize=(12,2))
#                 plt.imshow(attn_w_np[0:1], cmap='viridis', aspect='auto')
#                 plt.colorbar()
#                 plt.xlabel("Sequence Position")
#                 plt.ylabel("Batch")
#                 plt.title("Attention weights (target-based pooling)")
#                 plt.show()



            
#     # batch 합치기
#     all_labels = np.concatenate(all_labels)
#     all_preds = np.concatenate(all_preds)
#     all_preds_bin = (all_preds > 0.5).astype(int)
    
#     f1 = f1_score(all_labels, all_preds_bin)
#     recall = recall_score(all_labels, all_preds_bin)
#     precision = precision_score(all_labels, all_preds_bin)
#     auc = roc_auc_score(all_labels, all_preds)
#     # ===================================================================
#     avg_val_loss = val_loss / val_samples

#     return avg_val_loss, f1, recall, precision, auc

# def focal_loss(logits, targets, alpha=0.8, gamma=2): # 알파는 1에 대한 가중치, 감마는 쉬운 샘플에 대한 손실 감소
#     probs = torch.sigmoid(logits)
#     ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
#     p_t = probs * targets + (1 - probs) * (1 - targets)
#     loss = alpha * (1 - p_t) ** gamma * ce_loss
#     return loss.mean()

# """ Training Loop with AMP & Gradient Accumulation """
# # 이 부분은 위에 모델 좀 고치고 난 다음에 학습 잘 안되면 적용해보던가 하자
# # Warmup scheduler
# # Cosine LR decay
# # AdamW optimizer 얘네 세개 나중에 정리하자 일단은 focal이랑 bce 테스트부터
# def train_model(PATHS, FEATURE_COLS, VOCABS, CFG, device='cpu', augmentations=None):
#     torch.cuda.empty_cache()
    
#     train_loader = get_data_loader(
#         PATHS['TRAIN_PATH'], 
#         FEATURE_COLS['NUMERIC'], 
#         FEATURE_COLS['CATEGORICAL'], 
#         FEATURE_COLS['SEQUENCE'], 
#         FEATURE_COLS['TARGET'], 
#         VOCABS['SEQUENCE'],
#         VOCABS['CATEGORICAL'],
#         CFG['CHUNK_SIZE'],
#         CFG['BATCH_SIZE'],
#         val=False,
#         shuffle=False, # 아나 생각해보나 IterableDataset이라 shuffle 못하잖아?
#         # augmentations=augmentations  # 추가적인 augmentation
#     )
#     val_loader = get_data_loader(
#         PATHS['TRAIN_PATH'], 
#         FEATURE_COLS['NUMERIC'], 
#         FEATURE_COLS['CATEGORICAL'], 
#         FEATURE_COLS['SEQUENCE'], 
#         FEATURE_COLS['TARGET'], 
#         VOCABS['SEQUENCE'],
#         VOCABS['CATEGORICAL'], 
#         CFG['CHUNK_SIZE'],
#         CFG['BATCH_SIZE'],
#         val=True,
#         shuffle=False
#     )

#     """
#     Loss Scaling (스케일링)
#     Loss Scaling은 mixed precision training에서 흔히 사용되는 기법인데, 
#     이걸 통해 모델 학습 속도를 높이고 메모리 사용을 줄일 수 있습니다. 
#     mixed precision training에서는 float16 형식을 사용해서 계산을 효율적으로 하죠.
    
#     1. Gradient Accumulation과 Loss Scaling:
#     Gradient Accumulation: 학습할 때 한 번에 전체 배치를 처리하는 대신, 
#     여러 개의 작은 배치에서 gradient를 누적시켜서 최종적으로 한번에 update하는 방식입니다. (즉, 역전파.. 학습용)
#     이는 GPU 메모리가 부족할 때 유용하고, 메모리 제약을 완화하는 데 도움을 줍니다.
#     이때, loss를 accumulation_steps로 나누는 이유는 작은 배치에서 계산된 loss가 실제로는 큰 배치의 loss가 되도록 맞추기 위함입니다.
#     - 예를 들어, accumulation_steps=4일 경우, 각 배치에서 계산된 loss를 4번 쌓은 후에 한 번에 모델 파라미터를 업데이트하죠.
#     - Loss 보정: Gradient Accumulation을 사용하면 손실 값이 계속 쌓이기 때문에, 그 값이 커질 수 있습니다. 그래서 각 배치에서 계산된 loss를 accumulation_steps로 나누어서 이 값을 보정하는 방식입니다.

#     2. Scaler (스케일러):
#     scaler.scale(loss)는 mixed precision에서의 손실 스케일링을 처리합니다. float16에서 계산하면 숫자 범위가 작아서 NaN이 발생할 수 있는데, 
#     이를 방지하려고 loss를 일시적으로 float32로 변환시켜서 계산해주는 역할을 합니다. scaler.step()은 계산된 gradient를 모델에 적용하는 단계입니다.
#     예시: scaler.scale(loss).backward()  # loss에 대해 backward pass 수행
#     - scaler.scale(loss)는 loss 값을 스케일링하고, backward()는 역전파를 통해 gradient를 계산합니다.
#     - 그 후, scaler.step(optimizer)는 계산된 gradient를 기반으로 옵티마이저로 가서 파라미터를 업데이트합니다.
#     따라서 loss / accumulation_steps가 맞는지 확인하려면, accumulation_steps만큼 gradient가 누적된 후에 손실을 보정하는지 체크해야 합니다.
#     """
#     # Model placeholder  스케쥴러 쓸까말까쓸까말까
#     if os.path.exists(PATHS['BEST_MODEL']):
#         print("model exist!")
#         checkpoint = torch.load(PATHS['BEST_MODEL'], map_location=device)
        
#         model.load_state_dict(checkpoint["model_state_dict"])
#         model.to(device)
        
#         optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
#         scaler = GradScaler()
#         scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
#         best_auc = checkpoint.get("best_auc", 0.0)
#         print(f"Best AUC: {checkpoint['best_auc']} | Valid Loss: {checkpoint['val_loss']}")
    
#     else:
#         print("model doesn't exist")
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
#         scaler = GradScaler()
#         best_auc = 0.0

#     """
#     BCE or Focal loss를 쓰면.. 불균형한 데이터에서 rare pos에 더 집중 가능!!

#     BCE는 가중치 더 주는 방식인가봐
#     but!!!! 실제 Positive 비율에 2프로긴 하지만... BCE에 51을 그대로 넣으면
#     Pos에 너무 많은 가중치를 줘서 걍 1로 예측 많이 해버린대...

#     Focal loss는 가중치가 변동되나봐.. 잘 맞춘 샘플에 대해 손실 줄이고
#     잘못 맞춘 샘플에 대해 더 큰 손실을 준대!!
#     but!!! 얘는 또 출력이 확률 형식이 아니라 뭔 문제가 생기나본데?
#     """
#     # criterion는 항상 새로 생성
#     pos_weight = torch.tensor(51.42546463012695, device=device) # 51.42546463012695
#     # pos_ratio = y.sum() / len(y)
#     # pos_weight = (1 - pos_ratio) / pos_ratio 이건 각 청크마다 pos 비율이 다르니까 그거 하는겨
#     # pos_weight = torch.tensor([pos_weight], device=device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     # criterion = focal_loss
    
#     # total_steps = ((TOTAL_SAMPLES - 3*CHUNK_SIZE)) * SAMPLE_RATIO // CFG['BATCH_SIZE']
#     v = (CFG['TOTAL_SAMPLES']//CFG['CHUNK_SIZE']//10 + 1)*CFG['CHUNK_SIZE']
#     train_steps = ((CFG['TOTAL_SAMPLES'] - v)) * CFG['SAMPLE_RATIO'] // CFG['BATCH_SIZE']
    
#     for epoch in range(1, CFG['EPOCHS']+1):
#         # Train
#         avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, 
#                                      CFG['ACCUMULATION_STEPS'], device, train_steps, epoch)
#         # Validation
#         avg_val_loss, f1, recall, precision, auc =  valid_epoch(model, val_loader, criterion, device, epoch)

#         print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
#         print(f"Val F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | AUC: {auc:.4f}")
        
#         # CTR에선 평가지표로 AUC를 젤 많이 쓴대! 긍까 이걸 기준으로 모델 저장하자
#         if auc > best_auc:
#             best_auc = auc

#             checkpoint = {
#                 "model_state_dict": model.state_dict(), # 얘가 모델 파라미턴가보다
#                 "epoch": epoch,
#                 "best_auc": auc,
#                 "val_loss": avg_val_loss,
#                 "train_loss": avg_train_loss,
#                 "f1": f1,
#                 "recall": recall,
#                 "precision": precision,
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scaler_state_dict": scaler.state_dict()
#                 # "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
#             }
                        
#             torch.save(checkpoint, f"{PATHS['SAVE']}/best_{CFG['SAMPLE_RATIO']}_auc.pt")
#             print(f"✅ Model & metadata saved (AUC={auc:.4f})")

#         torch.save(model.state_dict(), f"{PATHS['SAVE']}/last_{CFG['SAMPLE_RATIO']}.pt")        
#         print("=================================================")
#     # return model # 추론 떄 학습한 모델 바로 가져다 쓸 수 있으니까