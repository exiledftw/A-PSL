# SANA Sign — Developer Handoff Document
### For: Any developer continuing this project after SIMPACT 2026

> **Repo:** `exiledftw/A-PSL` (local: `E:\sign-language\Beta`)
> **Last Updated:** September 10, 2026
> **Written by:** Rehan + Khizer

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Critical Contracts — Do Not Break These](#3-critical-contracts--do-not-break-these)
4. [The AI Model](#4-the-ai-model)
5. [The Dataset](#5-the-dataset)
6. [Inference Pipeline Step by Step](#6-inference-pipeline-step-by-step)
7. [Training Pipeline Step by Step](#7-training-pipeline-step-by-step)
8. [The Avatar System](#8-the-avatar-system)
9. [Known Issues and Pitfalls](#9-known-issues-and-pitfalls)
10. [Environment Setup](#10-environment-setup)
11. [Key Scripts Reference](#11-key-scripts-reference)
12. [Glossary](#12-glossary)

---

## 1. Project Overview

SANA Sign is a two-pipeline system:

- **Pipeline A (Patient → Doctor):** Webcam captures PSL gesture → MediaPipe extracts keypoints → AI model translates → English + Urdu text displayed.
- **Pipeline B (Doctor → Patient):** Doctor speaks → Whisper STT → NLP phrase match → SANA avatar performs PSL sign in browser.

The AI model is a custom PyTorch model called `SANA_PSL_Translator` combining a Conv1D gesture tokenizer, a 2-layer Transformer encoder, and Google's `mT5-small` multilingual decoder.

---

## 2. Repository Structure

```
E:\sign-language\Beta\   (exiledftw/A-PSL)
│
├── webcam_inference.py          ← MAIN ENTRY POINT for Pipeline A (live demo)
├── record_medical_dataset.py    ← Records new training keypoints from webcam
├── text_to_gloss_parser.py      ← NLP phrase matcher for Pipeline B
│
├── step1_extract_pose.py        ← Pipeline B1: extracts 3D pose from video
├── step2_make_fbx.py            ← Pipeline B1: Blender script bakes FBX
│
├── avatar_player.html           ← Three.js FBX avatar player (browser)
├── avatar_bvh_player.html       ← Three.js BVH avatar player (browser)
│
├── Y Bot.fbx                    ← Mixamo humanoid skeleton (base mesh)
│
├── dataset/                     ← Training keypoints (15 classes × 60 files)
│   ├── Assalam_o_alaikum/       ← 60 × .npy files, each shape (60, 208)
│   ├── Yes/
│   ├── No/
│   └── ... (15 folders total)
│
├── medical_dataset/             ← FBX animation output files
├── bvh_files/                   ← DeepMotion BVH animations
├── pose_json/                   ← MediaPipe JSON from step1_extract_pose.py
│
├── memory/                      ← Daily AI session context logs (important history)
├── project docs/                ← All documentation lives here
│   ├── SANA_Sign_SIMPACT_Report.md
│   ├── SANA_Sign_Developer_Handoff.md   ← THIS FILE
│   ├── SANA_Sign_Team_Report.md
│   ├── architecture.md
│   ├── phases.md
│   ├── prd.md
│   ├── design.md
│   └── rules.md
│
└── sana_psl_medical_finetuned.pt  ← FINAL MODEL (~2.1 GB, do not delete)
```

---

## 3. Critical Contracts — Do Not Break These

These are the rules the model was trained with. If you change any of these without retraining, the model will give wrong predictions.

### 3.1 Feature Vector Layout (208 dims per frame)

```
[  0 :  66 ]  →  33 pose landmarks × (x, y)          ← always present
[ 66 : 108 ]  →  21 LEFT hand landmarks × (x, y)      ← PRIMARY signal slot
[108 : 150 ]  →  21 RIGHT hand landmarks × (x, y)
[150 : 208 ]  →  Face — always 58 zeros               ← SANA standard
```

**Why LEFT hand is primary:** The training data was recorded with a front-facing (selfie) webcam. MediaPipe labels the real right hand as "Left" on a front camera. The recording script swaps handedness intentionally. So the signer's dominant right hand ends up in the left-hand slot `[66:108]`. This is correct and intentional — do not "fix" it.

### 3.2 Temporal Pipeline (Must match training exactly)

```
raw_frames (variable length)
    │
    ▼  resample_sequence(raw_frames, target_frames=60)
60 frames
    │
    ▼  pad_to_max_seq_len(frames, max_len=100, dim=208)
100 × 208 tensor  ←─ THIS is what the model receives
```

**Do NOT** skip the 60-frame resample step. The 1D Conv layers learned temporal frequencies assuming 60 frames of motion. If you pad directly without resampling, predictions will be wrong.

### 3.3 Mirror Correction (Always ON)

In `webcam_inference.py`, `MIRROR_CORRECTION = True` by default. This swaps left/right handedness detected by MediaPipe, compensating for the selfie camera flip. The training dataset was recorded with this same correction. **Toggle with `m` key during demo — default ON.**

### 3.4 Model Loading

The checkpoint `sana_psl_medical_finetuned.pt` is a **flat `OrderedDict`**, not a nested `{"model_state_dict": ...}` dict. Load it like this:

```python
checkpoint = torch.load("sana_psl_medical_finetuned.pt", map_location="cpu")
model.load_state_dict(checkpoint)
```

---

## 4. The AI Model

### 4.1 Class: `SANA_PSL_Translator` (in `webcam_inference.py`)

```
SANA_PSL_Translator
├── visual_encoder  ← UpgradedSpatialTemporalEncoder
│   ├── tokenizer   ← TemporalGestureTokenizer
│   │   ├── Conv1d(208→256, kernel=5, stride=2)
│   │   ├── BatchNorm1d + GELU
│   │   ├── Conv1d(256→512, kernel=5, stride=2)
│   │   └── BatchNorm1d + GELU
│   │   Output: (Batch, 25, 512) — 100 frames → 25 gesture tokens
│   │
│   ├── pos_encoder ← SinusoidalPositionalEncoding
│   └── transformer_encoder ← 2-layer, d_model=512, nhead=8, ffn=1024
│
└── mt5  ← AutoModelForSeq2SeqLM("google/mt5-small")
         ← Only cross-attention layers fine-tuned
         ← All other mT5 weights remain frozen
```

### 4.2 Inference Call

```python
# Correct way — beam generation with 2 hypotheses
outputs = model.mt5.generate(
    inputs_embeds=visual_embeds,
    num_beams=4,
    num_return_sequences=2,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=20,
)
top1 = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
top2 = tokenizer.decode(outputs.sequences[1], skip_special_tokens=True)

# Confidence from beam scores
s1, s2 = outputs.sequences_scores[0].item(), outputs.sequences_scores[1].item()
confidence = math.exp(s1) / (math.exp(s1) + math.exp(s2))
```

**Do NOT use candidate teacher-forcing** (computing `model.mt5(..., labels=candidate).loss` for each class). This method is biased by phrase length and will always favour longer or shorter phrases regardless of the actual sign. Use beam generation as shown above.

---

## 5. The Dataset

### 5.1 Location and Format

```
E:\sign-language\dataset\
├── Assalam_o_alaikum\          60 .npy files
├── Yes\                        60 .npy files
├── No\                         60 .npy files
├── Mein beemar hu\             60 .npy files
├── Mujhay bukhar hai\          60 .npy files
├── Mere sarr mein dard hai\    60 .npy files
├── Meri aankh surkh hai\       60 .npy files
├── Mujhay chakkar aa rhy hein\ 60 .npy files
├── Mujhay dard kam hai\        60 .npy files
├── Mujhay dard tez hai\        60 .npy files
├── Ambulance ko call karro\    60 .npy files
├── There_has_been_an_accident\ 60 .npy files
├── is_blood_pressure_high_or_low\ 60 .npy files
├── Test_are_cheap_here\        60 .npy files
└── Mujhay kuch dawa khareedni hai\ 60 .npy files
```

Each `.npy` file: shape `(60, 208)` — 60 frames of 208-dim feature vectors.  
Files are **keypoints only** — no video stored.

### 5.2 How the Dataset Was Made

1. **Video recording** — Khizer and Rehan recorded PSL signs on laptop webcam using `record_medical_dataset.py`
2. **Keypoint extraction** — `keypoints.ipynb` (Google Colab) ran MediaPipe on each video → extracted 208-dim vectors → resampled to 60 frames
3. **6× augmentation** — spatial zoom, position shift, temporal jitter, coordinate noise → each raw recording became 6 training samples
4. **Output** — `.npy` files saved per class, uploaded to Kaggle for training

### 5.3 Recording New Phrases (if you need to add more)

```bash
python record_medical_dataset.py --label "New Phrase Label" --num_samples 10
```

This saves raw `.npy` keypoints. Then run `keypoints.ipynb` on Colab to augment and add to the dataset before retraining.

---

## 6. Inference Pipeline Step by Step

### Running the Live Demo

```bash
cd E:\sign-language\Beta
python webcam_inference.py
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Start / Stop recording |
| `m` | Toggle mirror correction (default ON) |
| `d` | Run collapse diagnostic |
| `q` | Quit |

### What Happens Internally

1. Webcam opens, MediaPipe PoseLandmarker + HandLandmarker initialise
2. User presses SPACE → frames start buffering
3. MediaPipe extracts 208-dim vector every frame (~17.2 FPS on CPU)
4. User presses SPACE again → recording stops
5. Raw buffer → `resample_sequence(raw, target=60)` → `pad_to_max_seq_len(..., 100, 208)`
6. Tensor fed to `SANA_PSL_Translator.visual_encoder` → visual embeddings
7. `model.mt5.generate(num_beams=4, num_return_sequences=2)` → top-1 and top-2 phrases
8. Confidence computed from beam scores
9. If confidence ≥ 85%: display English + Urdu. If not: show "Human Interpreter Required"
10. Result shown in HUD overlay on webcam feed

---

## 7. Training Pipeline Step by Step

### Phase 1 — ASL Foundation (already done)

Trained `UpgradedSpatialTemporalEncoder` + frozen `mT5-small` on How2Sign dataset (31k ASL clips) on Kaggle T4.  
Notebook: `A-PSL_Phase1-2-3.ipynb`

### Phase 2 — PSL Domain Adaptation (already done)

5-epoch few-shot fine-tune on 71-word PSL dataset (mohib123456/Kaggle).  
Val Loss: 4.62 → **0.73** (84% reduction).  
Notebook: `SANA_PSL_FewShot_Train.ipynb`

### Phase 3 — Medical Fine-Tuning (already done)

30-epoch fine-tune on 15-class medical keypoint dataset.  
Loss: 4.40 → **0.0315**  
Notebook: `E:\sign-language\backup\notebook399144a5d3.ipynb`  
Output: `sana_psl_medical_finetuned.pt`

### If You Need to Retrain

1. Upload updated dataset to Kaggle
2. Open `notebook399144a5d3.ipynb` on Kaggle
3. Point it to the updated dataset path
4. Run all cells — training runs 30 epochs
5. Download `sana_psl_medical_finetuned.pt` and replace the local copy
6. `webcam_inference.py` loads it automatically on next run

---

## 8. The Avatar System

### Pipeline B1 — Custom MediaPipe → FBX

```bash
# Step 1: Extract 3D pose from reference video
python step1_extract_pose.py --video "path/to/sign.MOV"
# Outputs: pose_json/sign_pose.json

# Step 2: Bake into FBX (requires Blender installed)
blender --background --python step2_make_fbx.py -- --pose pose_json/sign_pose.json
# Outputs: medical_dataset/sign.fbx
```

### Playing Animations in Browser

```bash
# Serve the repo locally
python -m http.server 8000
# Then open: http://localhost:8000/avatar_player.html
```

The player loads `Y Bot.fbx` as the skeleton, then plays any `.fbx` or `.bvh` animation on it.

### Three.js Known Issues (Already Fixed)

| Issue | Fix Applied |
|---|---|
| `fixDuplicateSkeleton()` needed | Y Bot.fbx has dual skeleton — fixed in avatar_player.html |
| `mixamorig:` prefix stripping | Track name matching fix in avatar_player.html |
| BVH retargeting formula | `finalQuat = bvhQuat × restQuat` — applied in BVH player |

---

## 9. Known Issues and Pitfalls

| Issue | Detail | Status |
|---|---|---|
| **Single signer only** | Model was trained on Khizer + Rehan only. May not generalise to other signers. | Known limitation |
| **CPU-only webcam speed** | MediaPipe runs at ~17.2 FPS on CPU. GPU acceleration not yet enabled. | Acceptable for demo |
| **15 phrases only** | The fine-tuned model only recognises these 15 medical phrases. Any other sign → wrong prediction. | By design for MVP |
| **"Test are cheap here" fallback** | The model defaults to this when input is all zeros (no hand detected). Expected behavior. | Intentional fallback |
| **Mirror must be ON** | If mirror correction is OFF on a front-facing camera, predictions will swap left/right and fail. | Documented |
| **Confidence threshold** | 85% threshold may suppress correct predictions if hands are partially occluded. | Known tradeoff |

---

## 10. Environment Setup

### Python Dependencies

```bash
pip install torch torchvision
pip install transformers sentencepiece
pip install mediapipe
pip install opencv-python
pip install Pillow arabic-reshaper python-bidi
pip install numpy
```

### Model Files Needed

| File | Path | Size |
|---|---|---|
| `sana_psl_medical_finetuned.pt` | `E:\sign-language\Beta\` | ~2.1 GB |
| `arial.ttf` (for Urdu rendering) | Windows system fonts | — |
| MediaPipe model files | Auto-downloaded on first run | ~10 MB |

---

## 11. Key Scripts Reference

| Script | Purpose | Run With |
|---|---|---|
| `webcam_inference.py` | Live PSL recognition demo | `python webcam_inference.py` |
| `record_medical_dataset.py` | Record new training keypoints | `python record_medical_dataset.py --label "Phrase"` |
| `text_to_gloss_parser.py` | NLP phrase matcher for Pipeline B | `python text_to_gloss_parser.py` |
| `step1_extract_pose.py` | Extract 3D pose from video | `python step1_extract_pose.py --video file.MOV` |
| `step2_make_fbx.py` | Bake pose JSON into FBX | Via Blender headless |
| `keypoints.ipynb` | Full keypoint extraction + augmentation | Google Colab |
| `notebook399144a5d3.ipynb` | Medical fine-tuning training | Kaggle |

---

## 12. Glossary

| Term | Meaning |
|---|---|
| PSL | Pakistani Sign Language |
| mT5 | Multilingual Text-to-Text Transfer Transformer (Google) |
| Conv1D | 1D Convolutional Neural Network — processes sequences |
| MediaPipe | Google library for real-time body landmark detection |
| VRM | Virtual Reality Model — 3D avatar format |
| FBX | Filmbox — 3D animation format (used for avatar animations) |
| BVH | Biovision Hierarchy — motion capture format |
| Beam search | Decoding strategy that explores multiple translation paths simultaneously |
| Mirror correction | Swapping left/right handedness to compensate for front-facing webcam flip |
| Resample | Stretching or compressing a sequence to a fixed number of frames |
| Feature vector | The 208-dim numerical representation of one frame of sign language |
| Fallback phrase | "Test are cheap here" — what the model outputs when given zero input (no signal) |
