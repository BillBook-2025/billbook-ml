# train.py
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import collate_fn, ClickIterableDataset
from config import CFG, PAD_IDX
import os

def train_model(model, TRAIN_PATH, feature_cols, seq_col, target_col,
                batch_size=CFG['BATCH_SIZE'], epochs=CFG['EPOCHS'],
                lr=CFG['LEARNING_RATE'], device='cuda', accumulation_steps=CFG['ACCUMULATION_STEPS'],
                pos_weight=None):

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(device='cuda')

    train_dataset = ClickIterableDataset(TRAIN_PATH, feature_cols, cat_cols=None, seq_col=seq_col, target_col=target_col, val=False)
    val_dataset = ClickIterableDataset(TRAIN_PATH, feature_cols, cat_cols=None, seq_col=seq_col, target_col=target_col, val=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_samples = 0.0, 0
        optimizer.zero_grad()
        for step, (x_num, x_cat, seq_items, _, target) in enumerate(train_loader):
            x_num, x_cat, seq_items, target = x_num.to(device), x_cat.to(device), seq_items.to(device), target.to(device)
            with autocast(device_type='cuda'):
                logits = model(x_num, x_cat, seq_items, None, target)
                loss = criterion(logits, target)/accumulation_steps
            scaler.scale(loss).backward()
            if (step+1)%accumulation_steps==0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item()*target.size(0)*accumulation_steps
            train_samples += target.size(0)
        avg_train_loss = train_loss/train_samples

        # validation
        model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for x_num, x_cat, seq_items, _, target in val_loader:
                x_num, x_cat, seq_items, target = x_num.to(device), x_cat.to(device), seq_items.to(device), target.to(device)
                with autocast(device_type='cuda'):
                    logits = model(x_num, x_cat, seq_items, None, target)
                    loss = criterion(logits, target)
                val_loss += loss.item()*target.size(0)
                val_samples += target.size(0)
        avg_val_loss = val_loss/val_samples
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("./save", exist_ok=True)
            torch.save(model.state_dict(),"./save/best.pt")
        torch.save(model.state_dict(),"./save/last.pt")
    return model