# --- Cell 2 ---
CONFIG = {
    # ═══════════════ PATHS ═══════════════
    "ZIP_PATH": "/kaggle/input/datasets/kkmalik/yt-asl/content",
    "TRANSLATIONS_PATH": "/kaggle/input/datasets/kkmalik/yt-asl-captions/YT.translations.all.json",
    "CHECKPOINT_DIR": "/kaggle/working/checkpoints",
    "LOG_DIR": "/kaggle/working/logs",

    # ═══════════════ MODEL ARCHITECTURE ═══════════════
    "MT5_MODEL_NAME": "google/mt5-small",        # Multilingual T5 (English + Urdu support)
    "D_MODEL": 512,                              # Transformer hidden dimension (matches mT5-small's d_model)
    "NUM_ENCODER_LAYERS": 4,                     # Spatial-Temporal Transformer depth
    "NUM_HEADS": 8,                              # Multi-head attention heads
    "DIM_FEEDFORWARD": 2048,                     # FFN intermediate size
    "DROPOUT": 0.1,                              # Dropout rate
    "MAX_SEQ_LEN": 300,                          # Maximum frames per clip (pad/truncate)
    "NUM_KEYPOINTS": 104,                        # 33 pose + 21 left hand + 21 right hand + 29 selected face
    "KEYPOINT_DIM": 2,                           # X, Y coordinates (2D keypoints)
    "INPUT_DIM": 208,                            # NUM_KEYPOINTS * KEYPOINT_DIM (flattened per frame)

    # ═══════════════ TRAINING ═══════════════
    "BATCH_SIZE": 8,                             # Actual batch size per step
    "GRADIENT_ACCUMULATION_STEPS": 4,            # Effective batch size = 8 * 4 = 32
    "LEARNING_RATE": 3e-4,                       # AdamW learning rate
    "WEIGHT_DECAY": 0.01,                        # AdamW weight decay
    "WARMUP_STEPS": 500,                         # Linear warmup steps
    "MAX_EPOCHS": 5,                             # Training epochs (1 epoch = 1 full zip file pass)
    "MAX_TARGET_LENGTH": 128,                    # Max tokens for mT5 decoder output
    "NUM_WORKERS": 2,                            # DataLoader workers
    "FP16": True,                                # Mixed precision training

    # ═══════════════ LORA (Phase 4 — disabled for now) ═══════════════
    "LORA_RANK": 16,
    "LORA_ALPHA": 32,
    "LORA_TARGET_MODULES": ["q", "v"],
    "LORA_ENABLED": False,                       # Set True only in Phase 4

    # ═══════════════ LOGGING ═══════════════
    "LOG_EVERY_N_STEPS": 50,
    "SAVE_EVERY_N_STEPS": 500,
    "SEED": 42,

    # ═══════════════ KEYPOINT SCHEMA (MediaPipe topology + auto-detected ═══════════════
    # ═══════════════ key-name fallbacks — verify against Cell 3's printed output) ══════
    # Standard MediaPipe Pose landmark indices (0-32) for the shoulders/hips used by
    # SignSpace normalization. These are anatomical constants, not tunable.
    "LEFT_SHOULDER_IDX": 11,
    "RIGHT_SHOULDER_IDX": 12,
    "LEFT_HIP_IDX": 23,
    "RIGHT_HIP_IDX": 24,
    # 29 of the 468 MediaPipe Face Mesh indices: outer+inner lip contour (mouth shape is
    # a primary non-manual marker in ASL/PSL), eyebrows (negation/wh-question marking),
    # eye corners + nose tip (head orientation). Dropping the other 439 face points saves
    # VRAM while keeping the linguistically load-bearing face signal.
    "SELECTED_FACE_INDICES": [
        0, 1, 13, 14, 17, 33, 37, 39, 40, 61, 63, 66, 70, 78, 84, 105, 181,
        263, 267, 269, 270, 291, 293, 296, 308, 314, 334, 336, 405,
    ],
    # The LINDAT export's exact per-frame key names aren't pinned down by the dataset
    # docs, so we try a primary name first and fall back to a known alternate. Cell 3
    # prints the real structure — if neither alternate matches, add the real key here.
    "POSE_LANDMARK_KEYS": ["pose", "pose_landmarks"],
    "LEFT_HAND_LANDMARK_KEYS": ["left_hand", "left_hand_landmarks"],
    "RIGHT_HAND_LANDMARK_KEYS": ["right_hand", "right_hand_landmarks"],
    "FACE_LANDMARK_KEYS": ["face", "face_landmarks"],
}

assert CONFIG["NUM_KEYPOINTS"] * CONFIG["KEYPOINT_DIM"] == CONFIG["INPUT_DIM"], \
    "INPUT_DIM must equal NUM_KEYPOINTS * KEYPOINT_DIM"
assert len(CONFIG["SELECTED_FACE_INDICES"]) == 29, "Expected exactly 29 selected face landmarks"


# --- Cell 4 ---
# sentencepiece backs the mT5 tokenizer and isn't always preinstalled on Kaggle's
# base image — installing it here keeps this notebook runnable with zero manual setup.
!pip install -q sentencepiece --no-input

# ═══════════════ Standard library ═══════════════
import os
import json
import zipfile
import random
import math
import time
import logging
import pathlib

# ═══════════════ Data science ═══════════════
import numpy as np

# ═══════════════ PyTorch core ═══════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# ═══════════════ PyTorch data ═══════════════
from torch.utils.data import Dataset, DataLoader, random_split

# ═══════════════ HuggingFace ═══════════════
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

# ═══════════════ Visualization / utilities ═══════════════
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display, HTML


def set_seed(seed: int) -> None:
    """
    Seeds every RNG the pipeline touches (Python, NumPy, PyTorch CPU + all CUDA
    devices) so a given CONFIG["SEED"] reproduces the same train/val split, model
    init, and dropout pattern run-to-run — important for comparing Phase 3 runs
    against each other and for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["SEED"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# --- Cell 6 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("health_check")


def describe_structure(obj, max_items=5, _depth=0, _max_depth=3):
    """
    Generic recursive introspection of an arbitrary JSON-loaded object — prints
    types, lengths, and dict keys without assuming any particular schema. This is
    what lets Cell 3 reveal the LINDAT export's *actual* layout, since the dataset
    docs don't pin down whether landmarks are nested lists, dicts, or something
    else (see CONFIG's KEYPOINT SCHEMA comment).
    """
    indent = "  " * _depth
    if _depth > _max_depth:
        print(f"{indent}... (truncated, max depth reached)")
        return
    if isinstance(obj, dict):
        print(f"{indent}dict with {len(obj)} keys: {list(obj.keys())[:max_items]}")
        for k in list(obj.keys())[:max_items]:
            print(f"{indent}['{k}'] ->")
            describe_structure(obj[k], max_items, _depth + 1, _max_depth)
    elif isinstance(obj, list):
        print(f"{indent}list with {len(obj)} items")
        if len(obj) > 0:
            print(f"{indent}item[0] ->")
            describe_structure(obj[0], max_items, _depth + 1, _max_depth)
    else:
        preview = str(obj)
        if len(preview) > 80:
            preview = preview[:80] + "..."
        print(f"{indent}{type(obj).__name__}: {preview}")


# ─── 1-2. Open the ZIP archive and list its contents ───────────────────────────
# This is the single most important sanity check in the notebook: if the ZIP_PATH
# is wrong or the archive is corrupt, everything downstream (Cells 5-11) wastes
# GPU time before failing, so we fail here, fast and cheap, instead.
with zipfile.ZipFile(CONFIG["ZIP_PATH"], "r") as zf:
    zip_namelist = zf.namelist()

print(f"Total files inside zip: {len(zip_namelist)}")
print("First 10 filenames:")
for name in zip_namelist[:10]:
    print(f"  {name}")

# ─── 3-4. Load translations and report basic counts ────────────────────────────
with open(CONFIG["TRANSLATIONS_PATH"], "r", encoding="utf-8") as f:
    translations = json.load(f)

total_clips = sum(len(v.get("clip_order", [])) for v in translations.values())
print(f"\nTotal video IDs in translations file: {len(translations)}")
print(f"Total clips in translations file:     {total_clips}")

# ─── 5. Read ONE sample keypoint JSON from inside the zip and inspect it ───────
# We deliberately do NOT assume a schema here — we print whatever is actually
# there. Compare this output against CONFIG["POSE_LANDMARK_KEYS"] etc. and update
# CONFIG if the real key names differ from our best-guess defaults.
sample_filename = zip_namelist[0]
with zipfile.ZipFile(CONFIG["ZIP_PATH"], "r") as zf:
    with zf.open(sample_filename) as fh:
        sample_json = json.load(fh)

print(f"\n{'='*70}\nSample file: {sample_filename}\n{'='*70}")
describe_structure(sample_json)

# ─── 6. Print the matching English translation for that clip ──────────────────
sample_clip_id = pathlib.Path(sample_filename).stem  # strip ".json"
matched_translation = None
for video_id, video_data in translations.items():
    entry = video_data.get(sample_clip_id)
    if entry is not None:
        matched_translation = entry.get("translation")
        break

print(f"\nClip ID:            {sample_clip_id}")
print(f"Matching translation: {matched_translation!r}")
if matched_translation is None:
    logger.warning(
        "No translation found for the sampled clip_id — this is fine if it just "
        "means that particular file isn't in the translations JSON, but if EVERY "
        "clip_id fails to match, the filename<->key convention has changed."
    )


# --- Cell 8 ---
def _get_first_present(d: dict, keys: list):
    """Returns d[key] for the first key in `keys` that exists in d, else raises."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(
        f"None of {keys} found in frame dict with keys {list(d.keys())}. "
        f"Update the matching *_LANDMARK_KEYS list in CONFIG (Cell 1) to match "
        f"the real key name printed by Cell 3."
    )


def _landmarks_to_array(landmark_list) -> np.ndarray:
    """
    Converts one body part's landmark list into an (N, 2) array of (x, y),
    accepting either of the two common MediaPipe-export conventions:
      - list of dicts:  [{"x": .., "y": .., "z": .., "visibility": ..}, ...]
      - list of lists:  [[x, y, ...], ...]
    Any extra fields (z, visibility) are dropped since CONFIG["KEYPOINT_DIM"] == 2.
    An empty list (hand not detected on this frame) becomes an (0, 2) array,
    zero-filled later by select_keypoints.
    """
    if len(landmark_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    first = landmark_list[0]
    if isinstance(first, dict):
        return np.array([[pt["x"], pt["y"]] for pt in landmark_list], dtype=np.float32)
    return np.array([[pt[0], pt[1]] for pt in landmark_list], dtype=np.float32)


def _extract_frames(raw_json) -> list:
    """
    Normalizes the top-level per-clip JSON into a plain list of per-frame dicts.
    Handles both a bare `[...]` list of frames and a `{"frames": [...]}` wrapper,
    since the exact LINDAT top-level layout isn't pinned down by the dataset docs
    (see Cell 3's health check).
    """
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict):
        if "frames" in raw_json:
            return raw_json["frames"]
        for value in raw_json.values():
            if isinstance(value, list) and len(value) > 0:
                return value
    raise ValueError(
        "Could not locate a frame list in this keypoint JSON. Inspect the Cell 3 "
        "output and update _extract_frames() to match the real top-level layout."
    )


def select_keypoints(raw_frame: dict) -> np.ndarray:
    """
    From one frame's full MediaPipe output, selects only the 104 landmarks we
    care about: 33 pose + 21 left hand + 21 right hand + 29 selected face
    (dropping the other 439 face-mesh points to save VRAM — see CONFIG's
    SELECTED_FACE_INDICES comment for which 29 and why).

    A hand that MediaPipe failed to detect on this frame (occluded, out of
    frame) arrives as an empty list; we zero-fill it rather than dropping the
    frame, so every frame keeps a fixed 104-landmark shape.

    Returns: np.array of shape (104 * 2,) — flattened [x1, y1, x2, y2, ...]
    """
    pose = _landmarks_to_array(_get_first_present(raw_frame, CONFIG["POSE_LANDMARK_KEYS"]))
    left_hand = _landmarks_to_array(_get_first_present(raw_frame, CONFIG["LEFT_HAND_LANDMARK_KEYS"]))
    right_hand = _landmarks_to_array(_get_first_present(raw_frame, CONFIG["RIGHT_HAND_LANDMARK_KEYS"]))
    face_full = _landmarks_to_array(_get_first_present(raw_frame, CONFIG["FACE_LANDMARK_KEYS"]))

    if left_hand.shape[0] == 0:
        left_hand = np.zeros((21, 2), dtype=np.float32)
    if right_hand.shape[0] == 0:
        right_hand = np.zeros((21, 2), dtype=np.float32)
    face_selected = face_full[CONFIG["SELECTED_FACE_INDICES"]]  # (29, 2)

    combined = np.concatenate([pose, left_hand, right_hand, face_selected], axis=0)  # (104, 2)
    assert combined.shape == (CONFIG["NUM_KEYPOINTS"], CONFIG["KEYPOINT_DIM"]), \
        f"Unexpected keypoint shape {combined.shape} — a landmark group returned " \
        f"the wrong count (pose={pose.shape}, lh={left_hand.shape}, " \
        f"rh={right_hand.shape}, face={face_selected.shape})."
    return combined.reshape(-1).astype(np.float32)  # (208,)


def normalize_signspace(keypoints_sequence: np.ndarray) -> np.ndarray:
    """
    SignSpace Normalization (Zelezny et al.):
      1. Center every frame on the midpoint between the left and right shoulders.
      2. Scale every frame by that frame's torso length (shoulder midpoint to hip
         midpoint).
      3. This makes the model invariant to signer height, distance from camera,
         and position in frame — without it, the encoder would have to waste
         capacity learning to normalize pose scale/position instead of learning
         sign structure.

    We normalize per-frame (not once per whole clip) so the model stays correct
    even if the signer moves closer to or further from the camera mid-clip; the
    trade-off is slightly more sensitivity to single-frame pose-estimation noise
    in the shoulders/hips. If that noise turns out to be a problem in practice, a
    per-clip median torso length is a reasonable variant to try instead.

    Input:  (T, K*2) array (flattened per-frame keypoints)
    Output: (T, K*2) array, normalized
    """
    T, flat_dim = keypoints_sequence.shape
    K = flat_dim // 2
    seq = keypoints_sequence.reshape(T, K, 2)  # (T, K, 2) — readable per-landmark indexing

    l_shoulder = seq[:, CONFIG["LEFT_SHOULDER_IDX"], :]
    r_shoulder = seq[:, CONFIG["RIGHT_SHOULDER_IDX"], :]
    l_hip = seq[:, CONFIG["LEFT_HIP_IDX"], :]
    r_hip = seq[:, CONFIG["RIGHT_HIP_IDX"], :]

    shoulder_mid = (l_shoulder + r_shoulder) / 2.0  # (T, 2)
    hip_mid = (l_hip + r_hip) / 2.0                 # (T, 2)

    torso_length = np.linalg.norm(shoulder_mid - hip_mid, axis=-1, keepdims=True)  # (T, 1)
    torso_length = np.clip(torso_length, a_min=1e-6, a_max=None)  # guard divide-by-zero

    centered = seq - shoulder_mid[:, None, :]
    normalized = centered / torso_length[:, None, :]

    return normalized.reshape(T, flat_dim).astype(np.float32)


def pad_or_truncate(sequence: np.ndarray, max_len: int):
    """
    Pads short sequences with zeros or truncates long sequences to max_len frames,
    so every clip in a batch has the same time dimension regardless of its
    original length.

    Returns: (max_len, K*2) array, actual_length (int, number of REAL frames
             before padding — used to build the attention mask)
    """
    T, flat_dim = sequence.shape
    if T >= max_len:
        return sequence[:max_len].astype(np.float32), max_len
    padded = np.zeros((max_len, flat_dim), dtype=np.float32)
    padded[:T] = sequence
    return padded, T


# --- Cell 10 ---
class YouTubeASLDataset(Dataset):
    """
    Zero-storage-footprint dataset that reads keypoint JSONs directly from a
    compressed ZIP archive and pairs them with English translations.

    CRITICAL DESIGN DECISIONS:
    1. We open the ZipFile in __getitem__, not __init__, because ZipFile objects
       are NOT safe to share across DataLoader worker processes.
    2. We build the clip_id index in __init__ by iterating the translations JSON
       (not by listing the zip contents — that would require reading the full
       central directory of a 34GB file for every worker process).
    3. We pre-tokenize all translations in __init__ to avoid redundant tokenizer
       calls during training.
    """

    def __init__(self, config: dict, logger: logging.Logger = None):
        self.config = config
        self.zip_path = config["ZIP_PATH"]  # string path, NOT an open ZipFile handle
        self.max_seq_len = config["MAX_SEQ_LEN"]
        self.logger = logger or logging.getLogger("YouTubeASLDataset")

        with open(config["TRANSLATIONS_PATH"], "r", encoding="utf-8") as f:
            translations_raw = json.load(f)

        # Flatten the nested {video_id: {clip_order, clip_id: {translation}}}
        # structure into a flat list of (clip_id, translation) pairs, walking
        # clip_order for deterministic iteration.
        self.samples = []  # list of (clip_id, translation_text)
        for video_id, video_data in translations_raw.items():
            for clip_id in video_data.get("clip_order", []):
                clip_entry = video_data.get(clip_id)
                if clip_entry is None or "translation" not in clip_entry:
                    continue
                self.samples.append((clip_id, clip_entry["translation"]))

        self.tokenizer = AutoTokenizer.from_pretrained(config["MT5_MODEL_NAME"])

        # Pre-tokenize every translation once up front so __getitem__ never calls
        # the tokenizer during training — tokenizing ~39k short strings is cheap
        # to do once, and this keeps DataLoader workers CPU-light per batch.
        self._tokenized_labels = {}
        for clip_id, text in self.samples:
            encoded = self.tokenizer(
                text,
                max_length=config["MAX_TARGET_LENGTH"],
                truncation=True,
                padding=False,  # dynamic padding happens later, in collate_fn
            )
            self._tokenized_labels[clip_id] = encoded["input_ids"]

        self.logger.info(
            f"YouTubeASLDataset indexed {len(self.samples)} clip_ids from translations "
            f"(not yet checked against the zip — missing files are skipped lazily in __getitem__)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # Not every clip_id in the translations JSON has a matching file in THIS
        # particular zip shard (HARD CONSTRAINT #8). Rather than crash or return
        # None (which would need a custom collate_fn null-check anyway), we walk
        # forward from idx until we find a sample that actually loads, wrapping
        # around the dataset once. This keeps every index always returning a
        # valid tensor, which is what plain PyTorch DataLoader/collate_fn expect.
        n = len(self.samples)
        for attempt in range(n):
            clip_id, translation = self.samples[(idx + attempt) % n]
            filename = f"{clip_id}.json"
            try:
                # A fresh ZipFile handle per call — see class docstring, decision #1.
                with zipfile.ZipFile(self.zip_path, "r") as zf:
                    with zf.open(filename) as fh:
                        raw_json = json.load(fh)
            except KeyError:
                continue  # filename not present in this zip shard — skip silently
            except (zipfile.BadZipFile, json.JSONDecodeError) as e:
                self.logger.warning(f"Corrupt zip entry for {clip_id}: {e} — skipping")
                continue

            try:
                frames = _extract_frames(raw_json)
                if len(frames) == 0:
                    continue
                per_frame = [select_keypoints(f) for f in frames]
                keypoints = np.stack(per_frame, axis=0)             # (T, 208)
                keypoints = normalize_signspace(keypoints)            # (T, 208)
                keypoints, actual_len = pad_or_truncate(keypoints, self.max_seq_len)
            except Exception as e:
                self.logger.warning(f"Failed to process keypoints for {clip_id}: {e} — skipping")
                continue

            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
            attention_mask[:actual_len] = 1.0

            return {
                "input_ids": torch.from_numpy(keypoints).float(),
                "attention_mask": torch.from_numpy(attention_mask).float(),
                "labels": torch.tensor(self._tokenized_labels[clip_id], dtype=torch.long),
                "clip_id": clip_id,
            }

        raise RuntimeError(
            "Scanned the entire dataset index and found no clip_id with a matching "
            "file in the zip. Check CONFIG['ZIP_PATH'] and CONFIG['TRANSLATIONS_PATH']."
        )


# --- Cell 12 ---
full_dataset = YouTubeASLDataset(config=CONFIG, logger=logging.getLogger("YouTubeASLDataset"))
tokenizer = full_dataset.tokenizer  # reused everywhere downstream (decoding, pad_token_id, ...)

n_total = len(full_dataset)
n_val = max(1, int(0.1 * n_total))
n_train = n_total - n_val

split_generator = torch.Generator().manual_seed(CONFIG["SEED"])
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=split_generator)


def collate_fn(batch: list) -> dict:
    """
    Dynamic padding: every sample already comes out of __getitem__ padded to the
    CONFIG["MAX_SEQ_LEN"] upper bound, but most batches don't contain any clip
    that long — so we trim the whole batch down to its longest ACTUAL clip
    (via the attention mask) before it ever reaches the GPU. Labels get the
    equivalent treatment: padded to this batch's longest translation, not
    CONFIG["MAX_TARGET_LENGTH"] globally. Both save real FLOPs on an average
    batch without changing what the model sees for any individual clip.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)            # (B, MAX_SEQ_LEN, INPUT_DIM)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)  # (B, MAX_SEQ_LEN)
    clip_ids = [b["clip_id"] for b in batch]

    batch_max_len = int(attention_mask.sum(dim=1).max().item())
    batch_max_len = max(batch_max_len, 1)  # never trim to zero-length
    input_ids = input_ids[:, :batch_max_len, :]
    attention_mask = attention_mask[:, :batch_max_len]

    max_label_len = max(len(b["labels"]) for b in batch)
    # -100 is HF's standard "ignore this position in the loss" label id.
    labels_padded = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["labels"]
        labels_padded[i, : len(seq)] = seq

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "clip_id": clip_ids,
    }


train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=True,
    collate_fn=collate_fn,
)

steps_per_epoch = len(train_loader) // CONFIG["GRADIENT_ACCUMULATION_STEPS"]
total_optimizer_steps = steps_per_epoch * CONFIG["MAX_EPOCHS"]

print(f"Total clips (have a translation): {n_total}")
print(f"Train clips:                      {n_train}")
print(f"Val clips:                        {n_val}")
print(f"Batches per epoch (train, raw):   {len(train_loader)}")
print(f"Optimizer steps per epoch:        {steps_per_epoch}  (grad. accumulation = {CONFIG['GRADIENT_ACCUMULATION_STEPS']})")
print(f"Total optimizer steps ({CONFIG['MAX_EPOCHS']} epochs): {total_optimizer_steps}")


# --- Cell 14 ---
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard (non-learned) sinusoidal positional encoding. We use the fixed
    sin/cos formulation rather than a learned embedding table because clip
    length varies a lot (a few frames to CONFIG["MAX_SEQ_LEN"]), and sinusoidal
    PE generalizes cleanly to any length up to max_len without needing extra
    trainable parameters that would only ever see the longer end of the range.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model) — buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class InputProjection(nn.Module):
    """Linear layer mapping flattened per-frame keypoints (INPUT_DIM) to D_MODEL."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SpatialTemporalEncoder(nn.Module):
    """
    The 'eyes' of the model. Converts a sequence of 2D keypoint frames into a
    sequence of D_MODEL-dimensional embeddings that the mT5 decoder can attend
    to via cross-attention.

    Architecture (following Zelezny et al.):
      1. Linear projection: (INPUT_DIM) -> (D_MODEL) per frame.
      2. Positional encoding: sinusoidal (not learned) — handles variable-length
         sequences without extra parameters.
      3. Transformer encoder: NUM_ENCODER_LAYERS layers of standard
         self-attention, so each frame's representation can incorporate context
         from the whole clip (a handshape mid-sign is ambiguous without the
         frames around it).
      4. Output: (B, T, D_MODEL), ready for cross-attention with mT5.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.input_projection = InputProjection(config["INPUT_DIM"], config["D_MODEL"])
        self.pos_encoding = SinusoidalPositionalEncoding(config["D_MODEL"], max_len=config["MAX_SEQ_LEN"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["D_MODEL"],
            nhead=config["NUM_HEADS"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"],
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["NUM_ENCODER_LAYERS"])

    def forward(self, keypoints: torch.Tensor, attention_mask: torch.Tensor):
        """
        keypoints:       (B, T, INPUT_DIM)
        attention_mask:  (B, T), 1 = real frame, 0 = padding

        Returns:
          encoder_hidden_states: (B, T, D_MODEL)
          attention_mask:        (B, T), unchanged — passed through for mT5's
                                  cross-attention masking in Cell 8.
        """
        x = self.input_projection(keypoints)
        x = self.pos_encoding(x)
        # nn.TransformerEncoder expects a boolean "True = ignore this position" mask.
        key_padding_mask = attention_mask == 0
        encoded = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return encoded, attention_mask


# --- Cell 16 ---
class SignLanguageTranslator(nn.Module):
    """
    Full model: Visual Encoder -> Projection -> Frozen mT5 Decoder.

    The projection layer bridges the Visual Encoder's output space
    (D_MODEL=512) to mT5's encoder hidden size. For mt5-small the two already
    match (both 512), so this starts out close to identity — but keeping it as
    an explicit learned layer means D_MODEL can diverge from mT5's hidden size
    in later phases (a bigger visual encoder, a different mT5 size) with no
    architecture changes needed here.

    During Phase 3 (this notebook):
      - mT5 is COMPLETELY FROZEN (requires_grad=False on every mT5 parameter).
      - Only the Visual Encoder and Projection Layer are trained.

    During Phase 4 (fine-tuning, not implemented here):
      - LoRA adapters get injected into mT5's Q and V attention matrices.
      - The Visual Encoder is unfrozen with a lower learning rate.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.visual_encoder = SpatialTemporalEncoder(config)

        # MT5ForConditionalGeneration is a thin subclass of T5ForConditionalGeneration
        # (identical forward pass — only model_type/config_class differ, used by the
        # Auto* class-resolution system). Loading the mt5-small checkpoint directly
        # into T5ForConditionalGeneration produces the same weights and behavior, so
        # we can use the simpler, already-imported class directly.
        self.mt5 = T5ForConditionalGeneration.from_pretrained(config["MT5_MODEL_NAME"])
        mt5_hidden_size = self.mt5.config.d_model

        self.projection = nn.Linear(config["D_MODEL"], mt5_hidden_size)

        # ═══ HARD CONSTRAINT #7: freeze mT5 completely in Phase 3 ═══
        if config["LORA_ENABLED"]:
            # Phase 4 placeholder — not implemented in this Phase 3 notebook.
            raise NotImplementedError(
                "LoRA fine-tuning is Phase 4. Set CONFIG['LORA_ENABLED'] = False to run this notebook."
            )
        for p in self.mt5.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        `input_ids` here is the KEYPOINT tensor (B, T, INPUT_DIM), not token ids —
        named input_ids only so it lines up with the field name the Dataset /
        collate_fn already produce. `labels` are tokenized target text ids with
        -100 in padding positions, exactly as HF's T5 expects for teacher forcing.
        """
        encoder_hidden_states, encoder_attention_mask = self.visual_encoder(input_ids, attention_mask)
        encoder_hidden_states = self.projection(encoder_hidden_states)  # (B, T, mt5_hidden)

        # Feed our own encoder's output straight into mT5's DECODER as
        # cross-attention memory, bypassing mT5's own text encoder entirely —
        # this "encoder swap" is what turns a text-to-text model into a
        # skeleton-to-text model. T5ForConditionalGeneration accepts a raw
        # (hidden_states,) tuple here and wraps it in BaseModelOutput internally;
        # `attention_mask` at this call site is used as the ENCODER attention
        # mask for cross-attention, since encoder_outputs is already provided.
        outputs = self.mt5(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        return outputs  # outputs.loss, outputs.logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **generate_kwargs):
        """Same encoder swap as forward(), routed through mT5's .generate() for inference."""
        encoder_hidden_states, encoder_attention_mask = self.visual_encoder(input_ids, attention_mask)
        encoder_hidden_states = self.projection(encoder_hidden_states)

        # .generate() (unlike forward()) requires encoder_outputs to already be a
        # proper ModelOutput — a raw tuple won't satisfy the `.last_hidden_state`
        # / dict-style access it uses internally once encoder_outputs is present.
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            **generate_kwargs,
        )


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


model = SignLanguageTranslator(CONFIG).to(device)
total_params, trainable_params, frozen_params = count_parameters(model)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)  <- Visual Encoder + Projection")
print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)  <- mT5 (Phase 3)")


# --- Cell 18 ---
os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["LOG_DIR"], "training.log")),
        logging.StreamHandler(),
    ],
    force=True,  # re-configure cleanly even if an earlier cell already called basicConfig
)
logger = logging.getLogger("train")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # only Visual Encoder + Projection (mT5 is frozen)
    lr=CONFIG["LEARNING_RATE"],
    weight_decay=CONFIG["WEIGHT_DECAY"],
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=CONFIG["WARMUP_STEPS"],
    num_training_steps=total_optimizer_steps,  # from Cell 6
)

scaler = GradScaler(enabled=CONFIG["FP16"])

CHECKPOINT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], "latest.pt")
EMERGENCY_CHECKPOINT_PATH = os.path.join(CONFIG["CHECKPOINT_DIR"], "emergency.pt")


def save_checkpoint(path: str, epoch: int, global_step: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": CONFIG,
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path} (epoch={epoch}, global_step={global_step})")


# ─── Checkpoint resumption: pick up automatically where a previous run left off ──
start_epoch = 0
global_step = 0
if os.path.exists(CHECKPOINT_PATH):
    logger.info(f"Found existing checkpoint at {CHECKPOINT_PATH} — resuming.")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    logger.info(f"Resumed from epoch={start_epoch}, global_step={global_step}")
else:
    logger.info("No checkpoint found — starting fresh.")


def run_train_step_with_oom_backoff(batch: dict, max_split: int = 8):
    """
    HARD CONSTRAINT #2: never exceed the T4's 15GB VRAM budget, and reduce
    batch size automatically if a batch doesn't fit, rather than crashing the
    whole run. On a CUDA OOM, frees the failed activations and retries the
    SAME batch split into progressively smaller micro-batches, each with its
    own forward+backward — every micro-batch's loss is weighted by its share
    of the full batch before backward(), so the total gradient contribution
    (and therefore GRADIENT_ACCUMULATION_STEPS' effective-batch-size math)
    matches what a single unsplit batch would have produced.

    If a micro-batch OOMs AFTER earlier micro-batches in the same split
    already called backward() successfully, this batch's gradient
    contribution is now an unrecoverable mix of two different split scales —
    rather than risk a silently wrong effective batch size, we zero the
    partial gradients and skip the rest of this batch (a rare, self-correcting
    loss of one accumulation window, not a silent correctness bug).

    Returns this batch's mean loss for logging, or None if the batch was skipped.
    """
    B = batch["input_ids"].size(0)
    split_factor = 1
    while split_factor <= max_split:
        chunk_size = max(1, math.ceil(B / split_factor))
        chunk_start = 0
        try:
            total_loss_value = 0.0
            for chunk_start in range(0, B, chunk_size):
                end = min(chunk_start + chunk_size, B)
                sub_ids = batch["input_ids"][chunk_start:end].to(device)
                sub_mask = batch["attention_mask"][chunk_start:end].to(device)
                sub_labels = batch["labels"][chunk_start:end].to(device)
                sub_weight = (end - chunk_start) / B  # this chunk's share of the full batch

                with autocast(enabled=CONFIG["FP16"]):
                    outputs = model(input_ids=sub_ids, attention_mask=sub_mask, labels=sub_labels)
                    loss = outputs.loss * sub_weight / CONFIG["GRADIENT_ACCUMULATION_STEPS"]
                scaler.scale(loss).backward()
                total_loss_value += outputs.loss.item() * sub_weight
            return total_loss_value

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise  # a real bug, not VRAM pressure — never swallow it
            torch.cuda.empty_cache()
            if chunk_start == 0:
                # Failed on this split's very first micro-batch, so nothing in
                # this attempt has called backward() yet — safe to just retry smaller.
                logger.warning(f"CUDA OOM at micro-batch size {chunk_size} — retrying at a smaller size.")
                split_factor *= 2
                continue
            logger.warning(
                f"CUDA OOM partway through a size-{chunk_size} split "
                f"({chunk_start}/{B} samples already processed at this split) — skipping the rest of this batch."
            )
            optimizer.zero_grad()
            return None

    logger.error("CUDA OOM even at micro-batch size 1 — skipping this batch entirely.")
    return None


@torch.no_grad()
def run_validation(num_examples_to_show: int = 3):
    """Computes mean val loss and decodes a few example translations for a spot-check."""
    model.eval()
    losses = []
    examples = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=CONFIG["FP16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(outputs.loss.item())

        if len(examples) < num_examples_to_show:
            n_show = min(num_examples_to_show - len(examples), input_ids.size(0))
            gen_ids = model.generate(
                input_ids=input_ids[:n_show],
                attention_mask=attention_mask[:n_show],
                max_length=CONFIG["MAX_TARGET_LENGTH"],
                num_beams=4,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            label_ids_for_decode = labels[:n_show].clone()
            label_ids_for_decode[label_ids_for_decode == -100] = tokenizer.pad_token_id
            golds = tokenizer.batch_decode(label_ids_for_decode, skip_special_tokens=True)
            for cid, gold, pred in zip(batch["clip_id"][:n_show], golds, preds):
                examples.append((cid, gold, pred))

    model.train()
    return float(np.mean(losses)) if losses else float("nan"), examples


train_loss_history = []   # [(global_step, avg_loss), ...]
val_loss_history = []     # [(global_step, val_loss), ...]
peak_vram_logged = False
training_start_time = time.time()

try:
    epoch_bar = tqdm(range(start_epoch, CONFIG["MAX_EPOCHS"]), desc="Epochs", position=0)
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        step_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['MAX_EPOCHS']}", position=1, leave=False)
        for step, batch in enumerate(step_bar):
            batch_loss = run_train_step_with_oom_backoff(batch)
            if batch_loss is not None:
                running_loss += batch_loss

            # VRAM monitoring: report peak allocation right after the very first
            # batch, since that's the earliest point a real OOM would show up.
            if not peak_vram_logged and device.type == "cuda":
                torch.cuda.synchronize()
                logger.info(f"VRAM after first batch: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                peak_vram_logged = True

            if (step + 1) % CONFIG["GRADIENT_ACCUMULATION_STEPS"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["LOG_EVERY_N_STEPS"] == 0:
                    elapsed = time.time() - training_start_time
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                    eta_min = ((total_optimizer_steps - global_step) / steps_per_sec / 60) if steps_per_sec > 0 else float("inf")
                    avg_loss = running_loss / (CONFIG["LOG_EVERY_N_STEPS"] * CONFIG["GRADIENT_ACCUMULATION_STEPS"])
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"step={global_step}/{total_optimizer_steps} loss={avg_loss:.4f} "
                        f"lr={current_lr:.2e} steps/s={steps_per_sec:.3f} ETA={eta_min:.1f}min"
                    )
                    train_loss_history.append((global_step, avg_loss))
                    step_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
                    running_loss = 0.0

                if global_step % CONFIG["SAVE_EVERY_N_STEPS"] == 0:
                    save_checkpoint(CHECKPOINT_PATH, epoch, global_step)
                    val_loss, examples = run_validation()
                    val_loss_history.append((global_step, val_loss))
                    logger.info(f"[validation] step={global_step} val_loss={val_loss:.4f}")
                    for cid, gold, pred in examples:
                        logger.info(f"  clip={cid} | gold: {gold!r} | pred: {pred!r}")

        # End-of-epoch checkpoint, independent of SAVE_EVERY_N_STEPS alignment.
        save_checkpoint(CHECKPOINT_PATH, epoch + 1, global_step)

except Exception:
    logger.error("Training crashed — saving an emergency checkpoint before re-raising.", exc_info=True)
    save_checkpoint(EMERGENCY_CHECKPOINT_PATH, epoch, global_step)
    raise

total_train_time_sec = time.time() - training_start_time
logger.info(f"Training complete in {total_train_time_sec / 60:.1f} minutes ({global_step} optimizer steps).")


# --- Cell 20 ---
model.eval()

n_show = min(5, len(val_dataset))
sample_indices = random.sample(range(len(val_dataset)), n_show)
sanity_batch = collate_fn([val_dataset[i] for i in sample_indices])

sanity_input_ids = sanity_batch["input_ids"].to(device)
sanity_attention_mask = sanity_batch["attention_mask"].to(device)
sanity_labels = sanity_batch["labels"].to(device)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=sanity_input_ids,
        attention_mask=sanity_attention_mask,
        max_length=CONFIG["MAX_TARGET_LENGTH"],
        num_beams=4,
    )

predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

label_ids_for_decode = sanity_labels.clone()
label_ids_for_decode[label_ids_for_decode == -100] = tokenizer.pad_token_id
ground_truths = tokenizer.batch_decode(label_ids_for_decode, skip_special_tokens=True)

# This is the notebook's proof of life: does anything resembling English come
# out the other end of a skeleton sequence? Exact match isn't expected this
# early — coherent-ish structure is the bar for Phase 3.
print("=" * 80)
for clip_id, gold, pred in zip(sanity_batch["clip_id"], ground_truths, predictions):
    print(f"Clip:         {clip_id}")
    print(f"Ground truth: {gold}")
    print(f"Prediction:   {pred}")
    print("-" * 80)


# --- Cell 22 ---
peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
final_train_loss = train_loss_history[-1][1] if train_loss_history else float("nan")
final_val_loss = val_loss_history[-1][1] if val_loss_history else float("nan")
total_params, trainable_params, frozen_params = count_parameters(model)

# Estimate full-epoch time from the observed per-step throughput, scaled up to
# the ~39,000-clip prototype shard described in the project context — useful
# for deciding batch size / epoch count before committing to a long Phase 2 run.
FULL_SHARD_CLIP_ESTIMATE = 39000
clips_per_optimizer_step = CONFIG["BATCH_SIZE"] * CONFIG["GRADIENT_ACCUMULATION_STEPS"]
if global_step > 0 and total_train_time_sec > 0:
    sec_per_clip = total_train_time_sec / (global_step * clips_per_optimizer_step)
    estimated_full_epoch_min = sec_per_clip * FULL_SHARD_CLIP_ESTIMATE / 60
else:
    estimated_full_epoch_min = float("nan")

print("=" * 80)
print("PHASE 3 TRAINING SUMMARY")
print("=" * 80)
print(f"Peak VRAM usage:            {peak_vram_gb:.2f} GB  (T4 budget: 15 GB)")
print(f"Total training time:        {total_train_time_sec / 60:.1f} minutes")
print(f"Optimizer steps completed:  {global_step}")
print(f"Final train loss:           {final_train_loss:.4f}")
print(f"Final val loss:             {final_val_loss:.4f}")
print(f"Trainable parameters:       {trainable_params:,}")
print(f"Frozen parameters:          {frozen_params:,}")
print(f"Est. time for 1 full epoch over ~{FULL_SHARD_CLIP_ESTIMATE:,} clips: {estimated_full_epoch_min:.1f} minutes")

if train_loss_history:
    fig, ax = plt.subplots(figsize=(8, 4))
    steps, losses = zip(*train_loss_history)
    ax.plot(steps, losses, label="train loss")
    if val_loss_history:
        v_steps, v_losses = zip(*val_loss_history)
        ax.plot(v_steps, v_losses, label="val loss", marker="o")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("Phase 3 training curve")
    ax.legend()
    plt.tight_layout()
    plt.show()


