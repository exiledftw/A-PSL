import json

def make_cell(source, cell_type="code"):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

cells = []

# Header Markdown
cells.append(make_cell("""# SANA Sign - How2Sign Training Pipeline
### SANA AI HIMS / SIMPACT 2026 Showcase
**Model:** Visual-Encoder (Spatial-Temporal Transformer) + Frozen Google mT5 Decoder  
**Dataset:** How2Sign Holistic Landmarks (31k Frontal Clips)  
**Tuned Hyperparameters:** Optimized via Optuna TPE Search""", "markdown"))

# Cell 1: Environment Setup & Configuration
cells.append(make_cell("""# ── Cell 1: Environment Setup & Config ─────────────────────────────────────────
import os, sys, random, math, time, gc, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

# Set reproducible seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

CONFIG = {
    # Dataset Paths
    "DATASET_ROOT": "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features",
    "TRAIN_CSV":    "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features/metadata/how2sign_realigned_train.csv",
    "VAL_CSV":      "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features/metadata/how2sign_realigned_val.csv",
    "TRAIN_DIR":    "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features/train/frontal",
    "VAL_DIR":      "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features/val/frontal",
    "OUTPUT_DIR":   "/kaggle/working",
    
    # Model Hyperparameters (Tuned from Optuna)
    "MT5_MODEL_NAME":       "google/mt5-small",
    "D_MODEL":              512,
    "NUM_HEADS":            8,
    "NUM_ENCODER_LAYERS":   2,
    "DIM_FEEDFORWARD":      1024,
    "DROPOUT":              0.108,
    "MAX_SEQ_LEN":          300,
    "MAX_TARGET_LEN":       128,
    "INPUT_DIM":            208,
    
    # Training Parameters
    "MAX_EPOCHS":           15,
    "BATCH_SIZE":           4,
    "GRAD_ACCUM_STEPS":     2,
    "LEARNING_RATE":        2.00e-4,
    "WEIGHT_DECAY":         6.1357e-4,
    "WARMUP_RATIO":         0.05,
    "FP16":                 False, # FP32 ensures numerical stability
    
    # Schema selection
    "SELECTED_FACE_INDICES": [
        0, 1, 13, 14, 17, 33, 37, 39, 40, 61, 63, 66, 70, 78, 84, 105, 181,
        263, 267, 269, 270, 291, 293, 296, 308, 314, 334, 336, 405
    ],
    "LEFT_SHOULDER_IDX":  11,
    "RIGHT_SHOULDER_IDX": 12,
    "LEFT_HIP_IDX":       23,
    "RIGHT_HIP_IDX":      24,
}

print("Config loaded successfully.")"""))

# Cell 2: Keypoint Utilities & Preprocessing
cells.append(make_cell("""# ── Cell 2: Keypoint Normalization & Preprocessing ───────────────────────────
def extract_104_keypoints(raw_npy, selected_face_indices):
    \"\"\"
    Extracts 104 2D landmarks (33 pose + 21 left + 21 right + 29 face) from (T, 543, 3)
    \"\"\"
    pose       = raw_npy[:, 0:33, :2]
    face_full  = raw_npy[:, 33:501, :2]
    left_hand  = raw_npy[:, 501:522, :2]
    right_hand = raw_npy[:, 522:543, :2]
    face_sel   = face_full[:, selected_face_indices, :]
    
    combined = np.concatenate([pose, left_hand, right_hand, face_sel], axis=1) # (T, 104, 2)
    return combined

def normalize_signspace(seq, ls_idx=11, rs_idx=12, lh_idx=23, rh_idx=24):
    \"\"\"
    Normalizes coordinates relative to shoulders and torso bounding box.
    seq: (T, 104, 2)
    \"\"\"
    T, K, _ = seq.shape
    ls = seq[:, ls_idx, :]
    rs = seq[:, rs_idx, :]
    lh = seq[:, lh_idx, :]
    rh = seq[:, rh_idx, :]
    
    smid = (ls + rs) / 2.0
    hmid = (lh + rh) / 2.0
    torso = np.linalg.norm(smid - hmid, axis=-1, keepdims=True)
    torso = np.clip(torso, 1e-6, None)
    
    centered = seq - smid[:, None, :]
    normalized = centered / torso[:, None, :]
    return normalized.reshape(T, -1).astype(np.float32) # (T, 208)

def pad_or_truncate(seq, max_len):
    \"\"\"
    Pads with zeros or truncates sequence to max_len.
    \"\"\"
    T, D = seq.shape
    if T >= max_len:
        return seq[:max_len], max_len
    padded = np.zeros((max_len, D), dtype=np.float32)
    padded[:T] = seq
    return padded, T

print("Preprocessing utilities ready.")"""))

# Cell 3: Dataset Class & Dataloader setup
cells.append(make_cell("""# ── Cell 3: How2Sign Dataset & Loader ─────────────────────────────────────────
class How2SignDataset(Dataset):
    def __init__(self, csv_path, npy_dir, tokenizer, config):
        self.config = config
        self.npy_dir = npy_dir
        self.tokenizer = tokenizer
        
        # Load CSV metadata
        sep = "\\t" if "\\t" in open(csv_path).readline() else ","
        df = pd.read_csv(csv_path, sep=sep)
        
        # Build valid sample list
        self.samples = []
        for _, row in df.iterrows():
            s_name = str(row["SENTENCE_NAME"]).strip()
            text   = str(row["SENTENCE"]).strip()
            file_name = f"{s_name}_holistic.npy"
            file_path = os.path.join(npy_dir, file_name)
            
            if os.path.exists(file_path) and len(text) > 0:
                self.samples.append((file_path, text))
                
        print(f"Loaded {len(self.samples)} valid samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, text = self.samples[idx]
        
        # Load npy landmarks
        raw = np.load(file_path, allow_pickle=True) # (T, 543, 3)
        landmarks_104 = extract_104_keypoints(raw, self.config["SELECTED_FACE_INDICES"])
        normalized = normalize_signspace(
            landmarks_104, 
            self.config["LEFT_SHOULDER_IDX"], 
            self.config["RIGHT_SHOULDER_IDX"],
            self.config["LEFT_HIP_IDX"],
            self.config["RIGHT_HIP_IDX"]
        )
        
        # NaN / Inf safety guard
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        padded_seq, valid_len = pad_or_truncate(normalized, self.config["MAX_SEQ_LEN"])
        
        # Create attention mask for valid frames
        attention_mask = np.zeros(self.config["MAX_SEQ_LEN"], dtype=np.float32)
        attention_mask[:valid_len] = 1.0
        
        # Tokenize target text for mT5
        tokens = self.tokenizer(
            text,
            max_length=self.config["MAX_TARGET_LEN"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tokens["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 # PyTorch ignores -100 in cross-entropy
        
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": labels
        }

tokenizer = AutoTokenizer.from_pretrained(CONFIG["MT5_MODEL_NAME"])

print("Building Datasets...")
train_dataset = How2SignDataset(CONFIG["TRAIN_CSV"], CONFIG["TRAIN_DIR"], tokenizer, CONFIG)
val_dataset   = How2SignDataset(CONFIG["VAL_CSV"], CONFIG["VAL_DIR"], tokenizer, CONFIG)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=True)

print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")"""))

# Cell 4: Model Architecture
cells.append(make_cell("""# ── Cell 4: Visual Encoder + mT5 Decoder Model ───────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SpatialTemporalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ffn_dim, dropout, max_len):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        x = self.proj(src)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(out)

class How2SignTranslator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_encoder = SpatialTemporalEncoder(
            input_dim=config["INPUT_DIM"],
            d_model=config["D_MODEL"],
            num_heads=config["NUM_HEADS"],
            num_layers=config["NUM_ENCODER_LAYERS"],
            ffn_dim=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"],
            max_len=config["MAX_SEQ_LEN"]
        )
        
        # Load and freeze mT5
        print(f"Loading pretrained backbone: {config['MT5_MODEL_NAME']}...")
        self.mt5 = AutoModelForSeq2SeqLM.from_pretrained(config["MT5_MODEL_NAME"])
        for param in self.mt5.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask, labels=None):
        key_padding_mask = (attention_mask == 0)
        encoder_hidden_states = self.visual_encoder(input_ids, src_key_padding_mask=key_padding_mask)
        
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        outputs = self.mt5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
        
    def generate(self, input_ids, attention_mask, max_length=128):
        key_padding_mask = (attention_mask == 0)
        encoder_hidden_states = self.visual_encoder(input_ids, src_key_padding_mask=key_padding_mask)
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        return self.mt5.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask, max_length=max_length)

model = How2SignTranslator(CONFIG).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized: Trainable parameters = {trainable_params:,} / {total_params:,}")"""))

# Cell 5: Training & Validation Loop
cells.append(make_cell("""# ── Cell 5: Full 15-Epoch Training Loop with Checkpointing ─────────────────────
steps_per_epoch = len(train_loader) // CONFIG["GRAD_ACCUM_STEPS"]
total_steps     = steps_per_epoch * CONFIG["MAX_EPOCHS"]
warmup_steps    = int(total_steps * CONFIG["WARMUP_RATIO"])

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["LEARNING_RATE"],
    weight_decay=CONFIG["WEIGHT_DECAY"]
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler(enabled=CONFIG["FP16"])
best_val_loss = float("inf")
save_path = os.path.join(CONFIG["OUTPUT_DIR"], "best_how2sign_model.pt")

print("=" * 80)
print(f"Starting How2Sign Training: {CONFIG['MAX_EPOCHS']} Epochs, Effective Batch Size {CONFIG['BATCH_SIZE'] * CONFIG['GRAD_ACCUM_STEPS']}")
print(f"Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")
print("=" * 80)

for epoch in range(CONFIG["MAX_EPOCHS"]):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with autocast(enabled=CONFIG["FP16"]):
            outputs = model(inputs, mask, labels)
            loss = outputs.loss / CONFIG["GRAD_ACCUM_STEPS"]
            
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue
            
        scaler.scale(loss).backward()
        running_loss += loss.item() * CONFIG["GRAD_ACCUM_STEPS"]
        
        if (step + 1) % CONFIG["GRAD_ACCUM_STEPS"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        if (step + 1) % 500 == 0:
            avg_step_loss = running_loss / (step + 1)
            lr_curr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{CONFIG['MAX_EPOCHS']}] | Step [{step+1}/{len(train_loader)}] | Train Loss: {avg_step_loss:.4f} | LR: {lr_curr:.2e}")

    train_loss = running_loss / len(train_loader)
    
    # ── Validation Phase ──
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(enabled=CONFIG["FP16"]):
                outputs = model(inputs, mask, labels)
            if not torch.isnan(outputs.loss):
                val_losses.append(outputs.loss.item())
                
    val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
    elapsed  = time.time() - start_time
    
    print("-" * 80)
    print(f"Epoch {epoch+1}/{CONFIG['MAX_EPOCHS']} Complete | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save Best Checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.visual_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": CONFIG
        }, save_path)
        print(f"--> Saved New Best Model Checkpoint to: {save_path} (Val Loss: {val_loss:.4f})")
    print("-" * 80)

print("\\n" + "=" * 80)
print(f"TRAINING COMPLETE! Best Validation Loss: {best_val_loss:.4f}")
print("=" * 80)"""))

# Cell 6: Sample Translation Inference (Showcase)
cells.append(make_cell("""# ── Cell 6: Live Translation Demo / Sample Evaluation ─────────────────────────
print("Loading Best Checkpoint for Inference Demonstration...")
checkpoint = torch.load(save_path, map_location=device)
model.visual_encoder.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("=" * 80)
print("TRANSLATION EXAMPLES (Ground Truth vs. Model Prediction)")
print("=" * 80)

with torch.no_grad():
    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        inputs = sample["input_ids"].unsqueeze(0).to(device)
        mask   = sample["attention_mask"].unsqueeze(0).to(device)
        raw_labels = sample["labels"].numpy()
        
        # Filter out -100 pad tokens to decode ground truth
        valid_label_ids = [idx for idx in raw_labels if idx != -100]
        ground_truth = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
        
        generated_ids = model.generate(inputs, mask, max_length=128)
        predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Sample #{i+1}:")
        print(f"  [Ground Truth]: {ground_truth}")
        print(f"  [Predicted]:    {predicted_text}")
        print("-" * 60)"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("e:/sign-language/Beta/SANA_How2Sign_Train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("SANA_How2Sign_Train.ipynb created successfully.")