# Architecture Document
# SANA Sign — System Architecture

> **Project:** SANA Sign (A-PSL)
> **Version:** Phase 6 MVP
> **Last Updated:** September 2026

---

## 1. System Overview

SANA Sign is a two-pipeline, browser-based system. It has no dedicated server — everything runs client-side (Three.js) or via pre-existing cloud services (Kaggle inference, Whisper STT).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SANA SIGN SYSTEM                             │
│                                                                     │
│  ┌──────────────────┐          ┌──────────────────────────────┐    │
│  │  PIPELINE A       │          │  PIPELINE B                  │    │
│  │  Patient → Doctor │          │  Doctor → Patient            │    │
│  │  (Gesture→Text)   │          │  (Voice→Avatar Animation)    │    │
│  └────────┬─────────┘          └──────────────┬───────────────┘    │
│           │                                    │                    │
│    Webcam Input                         Microphone Input            │
│    MediaPipe Extract                    Whisper STT                 │
│    SANA Model Classify                  NLP Phrase Match            │
│    Confidence Check                     Trigger FBX Animation       │
│    Display English+Urdu                 SANA Avatar Renders         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline A — Gesture to Text (Patient → Doctor)

### 2.1 Flow Diagram

```
Webcam Frame
     │
     ▼
┌────────────────────────────────┐
│  MediaPipe Tasks API           │
│  PoseLandmarker (lite)         │  ──→ 33 landmarks × (x,y) = 66 values
│  HandLandmarker (num_hands=2)  │  ──→ 42 L-hand + 42 R-hand = 84 values
│  Face slots: 58 zeros          │  ──→ Always 0 (SANA standard)
│                                │
│  Output: 208-dim vector/frame  │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Temporal Preprocessing        │
│  1. Resample raw frames → 60   │  (linear interpolation per-dim)
│  2. Zero-pad 60 → 100 frames   │  (model sees [100 × 208] tensor)
│  3. Exp. smoothing α=0.75      │  (applied only on non-zero frames)
│  4. Mirror correction toggle   │  (handles webcam L/R flip)
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  SANA PSL Translator           │
│  Architecture:                 │
│  Conv1D Temporal Tokenizer     │  stride=2, 300→75 gesture tokens
│  Cross-Attention Visual Enc.   │  Spatial-temporal transformer
│  mT5-small Decoder (LoRA)      │  Multilingual text generation
│                                │
│  Fine-tuned checkpoint:        │
│  sana_psl_medical_finetuned.pt │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Confidence Gate               │
│  Score ≥ 85% → Display output  │
│  Score < 85% → "Human needed"  │
└────────────────┬───────────────┘
                 │
                 ▼
       English + Urdu Text
       Displayed to Doctor
```

### 2.2 Key Technical Contracts

| Contract | Value | Source of Truth |
|---|---|---|
| Input shape | (100, 208) per sample | keypoints.ipynb |
| Landmark order | [66 pose, 42 L-hand, 42 R-hand, 58 zeros] | keypoints.ipynb |
| Target frames | 60 (resample), then pad to 100 | MedicalDataset class |
| Smoothing | Exponential, alpha=0.75, non-zero only | keypoints.ipynb |
| Mirror correction | Default ON (toggle with 'm' key) | webcam_inference.py |
| Collapse diagnostic | 'd' key — tests real vs zeros vs noise | webcam_inference.py |
| Fallback phrase | "Test are cheap here" = degenerate input | HANDOFF.md |

---

## 3. Pipeline B — Text to Avatar (Doctor → Patient)

### 3.1 Flow Diagram

```
Doctor Speaks
     │
     ▼
┌────────────────────────────────┐
│  Speech-to-Text                │
│  OpenAI Whisper (local/API)    │
│  Handles Urdu + English        │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Text-to-Gloss NLP             │
│  Fuzzy string match OR         │
│  Lightweight BERT embedding    │
│  Maps sentence → Phrase ID     │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Animation Lookup Table        │
│  phrase_id → .fbx / .bvh file  │
│  (Pre-baked PSL animations)    │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Three.js WebGL Renderer       │
│  avatar_player.html            │
│  Loads: Y Bot.fbx (skeleton)   │
│  Plays: matched animation clip │
│  Avatar: SANA (VRM 0.0)        │
└────────────────┬───────────────┘
                 │
                 ▼
       SANA Avatar performs
       the PSL sign visually
```

### 3.2 Animation Asset System

```
medical_dataset/
├── how_are_you_doing.fbx        ← Custom (MediaPipe → step2_make_fbx.py)
├── how_do_you_feel_clean.fbx    ← Khizer DeepMotion (80 leg curves stripped)
└── [future phrases].fbx

bvh_files/
├── I_was_born_deaf.bvh
├── do_you_feel_dizzy.bvh
├── how_do_you_feel.bvh
└── keep_your_feet_moisturized.bvh
```

### 3.3 Two FBX Animation Pipelines

```
PIPELINE B1 — Custom MediaPipe → FBX
Video (.MOV)
  └─→ step1_extract_pose.py   (MediaPipe Tasks API → pose_json/)
      └─→ step2_make_fbx.py   (headless Blender → medical_dataset/*.fbx)
          └─→ avatar_player.html (Three.js plays on Y Bot.fbx)

PIPELINE B2 — DeepMotion → FBX (Khizer's route)
Actor wears suit + records with iPhone
  └─→ DeepMotion cloud processes → .fbx with full finger capture
      └─→ clean_khizer_fbx.py   (strips leg jitter curves)
          └─→ avatar_player.html (Three.js plays on Y Bot.fbx)
```

---

## 4. Avatar Stack

```
VRoid Studio 2.14
  └─→ SANA.vrm  (VRM 0.0, ~33k polygons, 14 materials)
      ├── XR Animator (live webcam motion → VRM)
      │     https://sao.animetheme.com/XR_Animator.html
      ├── Warudo (Steam — desktop live capture)
      └── avatar_player.html (Three.js — pre-baked FBX playback)

Y Bot.fbx (Mixamo Humanoid Skeleton)
  ├── Bone naming: mixamorig:* prefix
  ├── fixDuplicateSkeleton() required in Three.js
  │     (remaps Alpha_Joints to Alpha_Surface inner bones)
  └── BVH Retargeting: finalQuat = bvhQuat × restQuat
```

---

## 5. ML Model Architecture

```
Input: [Batch × 100 × 208]
         │
         ▼
Conv1D Temporal Tokenizer
  Conv1d(in=208, out=256, kernel=3, stride=2)  → [Batch × 75 × 256]
  Conv1d(in=256, out=512, kernel=3, stride=2)  → [Batch × 37 × 512]
         │
         ▼
Visual Encoder (Spatial-Temporal Transformer)
  MultiHeadAttention over gesture token sequence
         │
         ▼
Projection Layer (Linear)
  Maps visual embeddings → mT5 latent dimension (512)
         │
         ▼
mT5-small Decoder (with LoRA adapters)
  Generates: English phrase / Urdu text
         │
         ▼
Output: Translation String
```

### Model Checkpoints
| File | Size | Description |
|---|---|---|
| sana_psl_complete_55mb.pt | 45.9 MB | Full model, all phases |
| sana_psl_light_48mb.pt | 91.9 MB | Lighter variant |
| sana_psl_medical_finetuned.pt | ~2.1 GB | Medical fine-tuned (definitive) |

---

## 6. Repository Structure

```
exiledftw/A-PSL  (local: E:\sign-language\Beta)
│
├── step1_extract_pose.py       ← Pipeline B1: MediaPipe extraction
├── step2_make_fbx.py           ← Pipeline B1: Blender FBX baker
├── webcam_inference.py         ← Pipeline A: Live inference
├── text_to_gloss_parser.py     ← Pipeline B: NLP phrase matching
│
├── avatar_player.html          ← Three.js FBX avatar player
├── avatar_bvh_player.html      ← Three.js BVH avatar player
├── avatar_rig_tester.html      ← Bone binding debugger
│
├── Y Bot.fbx                   ← Mixamo base skeleton (mesh)
├── 23_paani.fbx                ← Reference sign animation
│
├── medical_dataset/            ← Output FBX animations
├── pose_json/                  ← MediaPipe landmark JSON
├── bvh_files/                  ← DeepMotion BVH animations
├── dataset/                    ← NPY keypoint files (4 signs)
│
├── memory/                     ← Daily AI agent context logs
├── plans/                      ← Architecture strategy docs
├── project docs/               ← THIS FOLDER (PRD, arch, rules...)
│
├── A-PSL_Phase1-2-3.ipynb      ← Main training notebook
├── SANA_PSL_FewShot_Train.ipynb← Medical fine-tuning notebook
├── SANA_Optuna_Sweep.ipynb     ← Hyperparameter sweep
└── webcam_inference.py         ← Live inference entrypoint
```

---

## 7. Deployment Target

| Target | Description |
|---|---|
| SIMPACT Demo Machine | Local laptop, no internet required for avatar playback |
| Browser | Chrome (no install needed) |
| ML Inference | Local Python or Kaggle notebook |
| Avatar Renderer | http://localhost:8000/avatar_player.html |
| STT | Whisper local model (offline capable) |

---

## 8. Known Architecture Constraints

| Constraint | Implication |
|---|---|
| Y Bot.fbx has dual skeleton (Alpha_Joints + Alpha_Surface) | fixDuplicateSkeleton() MUST be in Three.js or avatar won't deform |
| BVH retargeting: bvhQuat × restQuat | Direct quaternion injection breaks arm orientation |
| FBX animations need mixamorig: prefix stripping | Three.js track name matching fails without this |
| mT5 uses greedy decoding | Collapse risk increases with more phrase classes |
| ~40 training videos | Model does not generalize to unseen signers yet |
