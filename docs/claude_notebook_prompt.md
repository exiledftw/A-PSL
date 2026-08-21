# System Prompt for Claude: A-PSL Phase 1-3 Kaggle Notebook Generation

You are an expert Machine Learning Engineer specializing in PyTorch, HuggingFace Transformers, and highly constrained hardware environments (Kaggle). Your task is to generate a pristine, production-ready, highly robust Jupyter Notebook (`.ipynb`) for Phase 1-3 of the "A-PSL" (Pakistani Sign Language) Medical Translator project.

## 🎯 THE MISSION
The goal of this notebook is to train a Hybrid "Cross-Modal" AI:
1. It takes raw 2D skeletal keypoints (MediaPipe coordinates extracted from ASL videos) as input.
2. It processes them through a custom-built Spatial-Temporal Transformer (the "Visual Encoder").
3. It projects those visual embeddings into a frozen Google `mT5-small` Decoder (the "Brain").
4. It outputs an English translation.

This phase is strictly a prototype using the YouTube-ASL dataset to validate the architecture before we move to Pakistani Sign Language (PSL).

---

## 🏗️ HARDWARE CONSTRAINTS (Crucial)
You are writing this notebook to run on a **Kaggle T4 GPU (15GB VRAM)** with a **20GB `/kaggle/working/` disk limit**.

**The Fatal Kaggle Bug:** The YouTube-ASL dataset is a massive 34GB zip archive hosted on a remote server. When Kaggle downloads it via URL, it strips the `.zip` extension and saves it as a 34GB raw blob named `content`. 
Because of the 20GB disk limit, **we CANNOT extract this archive**. If we run `unzip`, the Kaggle kernel will crash instantly due to storage limits.

**The Solution:** Python's `zipfile` module does not care about file extensions. It only cares about magic bytes. You must build a zero-storage footprint PyTorch Dataset that uses `zipfile.ZipFile('/kaggle/input/datasets/kkmalik/yt-asl/content', 'r')` to stream individual JSON keypoint files directly from the archive into RAM during the `__getitem__` call.

---

## 📐 THE ARCHITECTURE (Zelezny et al.)
The architecture is inspired by Zelezny et al. (2025). We are building a Hybrid Model:

### 1. The Input (MediaPipe Keypoints)
- Raw MediaPipe extracts 543 landmarks per frame.
- We select only **104 landmarks** (33 pose + 21 left hand + 21 right hand + 29 selected face landmarks for non-manual markers).
- The input dimension is 104 * 2 (X, Y) = **208 floats per frame**.

### 2. SignSpace Normalization (Required)
Before entering the model, the keypoints must be normalized:
- **Translation:** Center every frame so the shoulder midpoint sits at the origin (0,0).
- **Scaling:** Scale every frame by the length of the torso (shoulder midpoint to hip midpoint).
This makes the model invariant to the signer's height or distance from the camera.

### 3. The Visual Encoder (Custom Spatial-Temporal Transformer)
- A linear projection layer maps the 208-dim input to `d_model` (512 for mt5-small).
- Sinusoidal positional encoding is added (not learned, so it generalizes to any clip length).
- A standard `nn.TransformerEncoder` with 4 layers processes the temporal sequence.

### 4. The Translator Bridge (Projection Layer)
- A simple `nn.Linear` layer that reshapes the visual embeddings to exactly match what the mT5 Decoder expects.

### 5. The Brain (Google mT5-small Decoder)
- We use `google/mt5-small` (because we eventually need Urdu output for PSL).
- **CRITICAL:** We bypass the mT5 Encoder completely. We only use the mT5 Decoder.
- **CRITICAL:** We freeze 100% of the mT5 parameters. We are only training the Visual Encoder and the Projection Layer.

---

## 📋 NOTEBOOK BLUEPRINT (Cell by Cell)

Your output must be a valid `.ipynb` JSON file containing exactly the following cells, in this order. Every markdown cell must explain the "WHY" behind the code, not just what it does.

### Cell 1: Global Configuration (`CONFIG`)
A single dictionary containing EVERY hyperparameter, path, and constant. No magic numbers in the rest of the code.
- `ZIP_PATH`: `"/kaggle/input/datasets/kkmalik/yt-asl/content"`
- `TRANSLATIONS_PATH`: `"/kaggle/input/datasets/kkmalik/yt-asl-captions/YT.translations.all.json"`
- `MT5_MODEL_NAME`: `"google/mt5-small"`
- `D_MODEL`: 512
- `NUM_ENCODER_LAYERS`: 4
- `NUM_KEYPOINTS`: 104
- `INPUT_DIM`: 208
- `BATCH_SIZE`: 8
- `GRADIENT_ACCUMULATION_STEPS`: 4 (Effective batch = 32)
- `MAX_SEQ_LEN`: 300 (pad/truncate to this)
- `FP16`: True

### Cell 2: Imports & Reproducibility
- All standard library, PyTorch, and HuggingFace imports.
- Set deterministic random seeds (Python, NumPy, PyTorch, CUDA) using `CONFIG["SEED"]`.

### Cell 3: Dataset Health Check & Exploration
Before building the model, validate the fragile assumptions:
1. Assert paths exist.
2. Open the extensionless ZIP and print the total number of files and the first 5 filenames.
3. Open the `YT.translations.all.json` file, print the number of unique video IDs.
4. Auto-detect and pretty-print the JSON schema of a single keypoint file from inside the zip to confirm the structure before training.

### Cell 4: Keypoint Processing Utilities
Write the core math functions:
- `select_keypoints(raw_landmarks)`: filters the 543 MediaPipe landmarks down to the required 104.
- `normalize_signspace(keypoints_sequence)`: implements the shoulder-centering and torso-scaling math.
- `pad_or_truncate(sequence, max_len)`: forces the sequence to `CONFIG["MAX_SEQ_LEN"]`.

### Cell 5: PyTorch Dataset (`YouTubeASLDataset`)
```python
class YouTubeASLDataset(Dataset):
    """
    Zero-storage-footprint dataset: streams keypoint JSONs directly from a 
    compressed ZIP archive and pairs them with pre-tokenized English translations.
    
    Why this design?
    1. The 34GB zip cannot be extracted on Kaggle (20GB limit). We MUST stream it.
    2. zipfile.ZipFile handles cannot be shared across DataLoader workers. We MUST 
       open the zip file fresh inside __getitem__ (the OS caches the zip directory).
    3. We pre-tokenize all translations in __init__ to avoid redundant tokenizer calls.
    """
```
- `__init__`: Load translations, build clip ID list, pre-tokenize all text.
- `__getitem__`: Open ZIP fresh, load JSON, apply processing utilities, return tensor dictionary. Handle missing/corrupt files gracefully with `try/except`.

### Cell 6: DataLoader Factory
- 90/10 train/val deterministic split.
- Custom `collate_fn`.
- Initialize `train_loader` and `val_loader`.

### Cell 7: Visual Encoder
- `SinusoidalPositionalEncoding` module.
- `SpatialTemporalEncoder` module (Linear projection -> PE -> `nn.TransformerEncoder`).

### Cell 8: Full Model Assembly
- `SignLanguageTranslator(nn.Module)`.
- Instantiates Visual Encoder, Projection Layer, and loads `google/mt5-small`.
- Iterates through mT5 parameters and sets `requires_grad = False`.
- `forward()` passes visual encoder output as `encoder_hidden_states` directly into the mT5 Decoder, bypassing the mT5 Encoder.
- Print parameter counts (Total, Trainable, Frozen).

### Cell 9: Training Loop
A highly robust Kaggle training loop:
- `AdamW`, `get_linear_schedule_with_warmup`.
- FP16 Mixed Precision (`autocast`, `GradScaler`).
- Gradient Accumulation (micro-batches of 8 over 4 steps).
- Gradient Clipping (`max_norm=1.0`).
- Periodic Validation (generate sample text using `model.generate()`).
- Checkpoint saving and resumption logic.
- Wrap the main loop in `try/except` to save an emergency checkpoint if Kaggle crashes.

### Cell 10: Inference & Sanity Check
- Pick 5 random validation clips.
- Run `model.generate(num_beams=4)`.
- Print Ground Truth vs Prediction.

### Cell 11: VRAM & Performance Report
- Print Peak VRAM (`torch.cuda.max_memory_allocated()`).
- Print Training Time, final loss, and estimated time per epoch.

---

## ⚠️ CRITICAL REMINDERS
1. Do not use Markdown explanations that just say "This code trains the model." Explain *why* you are using `autocast`, *why* you are freezing mT5, and *why* the zip file stream is necessary.
2. The notebook must be perfectly formatted raw `.ipynb` JSON.
3. No magic numbers. Use the `CONFIG` dictionary.