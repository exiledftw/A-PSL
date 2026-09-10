# SANA Sign — Pakistani Sign Language Medical Communication System
### Formal Project Report | SIMPACT 2026 — CIME Karachi, September 17, 2026

> **Project Name:** SANA Sign (A-PSL)
> **Team:** Rehan (ML Lead / Avatar), Khizer (Motion Capture / Signer), Reyhan (Avatar Design)
> **Event:** SIMPACT 2026 — Social Impact Technology Showcase
> **Status:** MVP Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [AI Model Architecture](#6-ai-model-architecture)
7. [Training Results](#7-training-results)
8. [Live Inference Performance](#8-live-inference-performance)
9. [Avatar System — SANA](#9-avatar-system--sana)
10. [Safety Design](#10-safety-design)
11. [Success Metrics](#11-success-metrics)
12. [Team and Roles](#12-team-and-roles)
13. [Future Roadmap](#13-future-roadmap)

---

## 1. Executive Summary

**SANA Sign** is a real-time, AI-powered, bilingual communication bridge between deaf patients and doctors in Pakistani hospitals. It operates as a two-way system:

- **Patient → Doctor:** The patient performs a Pakistani Sign Language (PSL) gesture in front of a standard webcam. The system recognizes the sign in under 100ms and displays the English and Urdu translation to the doctor.
- **Doctor → Patient:** The doctor speaks in Urdu or English. The system transcribes the speech, matches it to a known phrase, and displays a 3D animated nurse avatar (SANA) performing the corresponding PSL sign.

The system runs entirely on a standard laptop — no internet required for inference — making it deployable at any hospital bedside.

---

## 2. Problem Statement

Pakistan has approximately **1 million deaf and hard-of-hearing individuals**. When a deaf patient visits a hospital, communication breaks down entirely:

- There are **no certified PSL interpreters** available at most hospitals
- There is **no assistive technology** for sign language communication in clinical settings
- **Life-threatening mistranslations** occur during emergency triage
- Total reliance on handwritten notes — slow, error-prone, and unusable during physical examinations

```
  Deaf Patient ──[PSL]──→      ???      ──→ Doctor (Urdu/English Speaker)
  Doctor (Urdu/English) ──→   ???       ──→ Deaf Patient
```

There is nothing in the middle. **SANA Sign fills this gap.**

---

## 3. Solution Overview

SANA Sign is a two-pipeline system that bridges both directions of communication:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          SANA SIGN SYSTEM                            │
│                                                                      │
│  ┌─────────────────────────┐      ┌──────────────────────────────┐   │
│  │   PIPELINE A             │      │   PIPELINE B                 │   │
│  │   Patient → Doctor       │      │   Doctor → Patient           │   │
│  │   PSL Gesture → Text     │      │   Voice → Avatar Sign        │   │
│  └────────────┬────────────┘      └──────────────┬───────────────┘   │
│               │                                  │                   │
│   Webcam captures PSL sign          Doctor speaks Urdu / English     │
│   MediaPipe extracts skeleton       Whisper transcribes speech        │
│   AI model classifies gesture       NLP fuzzy-matches phrase          │
│   English + Urdu text shown         SANA avatar performs PSL sign     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. System Architecture

### 4.1 Pipeline A — Gesture to Text (Patient → Doctor)

```
Webcam Frame
     │
     ▼
┌─────────────────────────────────┐
│  MediaPipe Tasks API            │
│  PoseLandmarker  → 33 × (x,y)  │  = 66 values
│  HandLandmarker  → 21L + 21R   │  = 84 values
│  Face slot       → 58 zeros    │  (SANA standard, always zero)
│  Output: 208-dim vector/frame  │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  Temporal Preprocessing         │
│  1. Resample raw frames → 60    │  (linear interpolation per-dim)
│  2. Zero-pad 60 → 100 frames    │  (model input tensor: 100 × 208)
│  3. Exponential smoothing α=0.75│  (applied on non-zero frames only)
│  4. Mirror correction (ON)      │  (corrects webcam left/right flip)
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  SANA PSL Translator (AI Model) │
│  Conv1D Temporal Tokenizer      │
│  Spatial-Temporal Transformer   │
│  mT5-small Multilingual Decoder │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  Confidence Gate                │
│  Score ≥ 85%  →  Show output   │
│  Score < 85%  →  Human needed  │
└─────────────────┬───────────────┘
                  │
                  ▼
        English + Urdu translation
        displayed to Doctor
```

### 4.2 Feature Vector Layout (208 dimensions per frame)

| Slot | Dimensions | Content |
|---|---|---|
| Pose | `[0 : 66]` | 33 body pose landmarks × (x, y) |
| Left Hand | `[66 : 108]` | 21 left hand landmarks × (x, y) |
| Right Hand | `[108 : 150]` | 21 right hand landmarks × (x, y) |
| Face | `[150 : 208]` | Always zero — SANA standard |

### 4.3 Pipeline B — Voice to Avatar (Doctor → Patient)

```
Doctor speaks in Urdu or English
     │
     ▼
OpenAI Whisper (local, offline-capable)
     │
     ▼
Fuzzy NLP phrase matching (thefuzz library)
     │
     ▼
Pre-baked PSL animation lookup table
     │
     ▼
SANA nurse avatar (VRoid VRM 0.0) performs
the matching PSL sign in browser (Three.js)
```

---

## 5. Dataset

### 5.1 Training Data Summary

| Dataset | Role | Size | Source |
|---|---|---|---|
| How2Sign | ASL Foundation pre-training | ~31,000 clips | Public (PSewmuthu/Kaggle) |
| PSL Isolated Words | PSL domain adaptation | 71 words / signs | mohib123456 (Kaggle) |
| PSL Medical Custom | Medical fine-tuning | 60 keypoint files × 15 classes | Self-recorded — Khizer + Rehan |

### 5.2 Medical Dataset — 15 Phrase Classes

| # | Phrase | Category |
|---|---|---|
| 1 | Assalam o Alaikum | Greeting |
| 2 | Yes | Response |
| 3 | No | Response |
| 4 | Mein beemar hu | Symptom |
| 5 | Mujhay bukhar hai | Symptom |
| 6 | Mere sarr mein dard hai | Symptom |
| 7 | Meri aankh surkh hai | Symptom |
| 8 | Mujhay chakkar aa rhy hein | Symptom |
| 9 | Mujhay dard kam hai | Symptom |
| 10 | Mujhay dard tez hai | Symptom |
| 11 | Ambulance ko call karro | Emergency |
| 12 | There has been an accident | Emergency |
| 13 | Is blood pressure high or low | Medical Query |
| 14 | Test are cheap here | Medical Query |
| 15 | Mujhay kuch dawa khareedni hai | Medical Query |

### 5.3 Recording Protocol

- Recorded by **Khizer** and **Rehan** as signers
- Each class: **60 keypoint files** (`.npy`), shape `(60, 208)` per file (keypoints only, no video stored)
- Captured using `record_medical_dataset.py` with MediaPipe Tasks API on laptop webcam
- **Selfie mirror correction** applied at recording time: front-facing camera handedness is swapped so the right hand maps to the left-hand feature slot, matching the training convention
- **6× data augmentation** applied during keypoint extraction: spatial zoom, position shift, temporal speed warping, and coordinate jitter

---

## 6. AI Model Architecture

### 6.1 SANA PSL Translator

```
Input Tensor: [Batch × 100 × 208]
         │
         ▼
┌──────────────────────────────────────┐
│  TemporalGestureTokenizer            │
│  Conv1d(208 → 256, kernel=5, s=2)   │
│  BatchNorm1d + GELU                  │
│  Conv1d(256 → 512, kernel=5, s=2)   │
│  BatchNorm1d + GELU                  │
│  Output: [Batch × 25 × 512]          │   100 frames → 25 gesture tokens
└─────────────────────┬────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Sinusoidal Positional Encoding      │
│  2-Layer Transformer Encoder         │
│  d_model=512, nhead=8, FFN=1024      │
│  Dropout=0.1                         │
└─────────────────────┬────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  mT5-small Decoder                   │
│  560M parameters total               │
│  Only cross-attention layers trained │
│  Outputs: English phrase / Urdu text │
└─────────────────────┬────────────────┘
                      │
                      ▼
           Translation String
```

### 6.2 Training Strategy

| Stage | Description | Method |
|---|---|---|
| Phase 1 | ASL Foundation | Full Visual Encoder training on How2Sign (31k clips) |
| Phase 2 | PSL Adaptation | 5-epoch few-shot fine-tune on 71-word PSL dataset |
| Phase 3 | Medical Fine-Tune | 30-epoch fine-tune on 15-class medical keypoint dataset |

**Frozen during medical fine-tuning:** All mT5 weights except cross-attention layers (`decoder.block[i].layer[1]`)  
**Hardware:** Kaggle T4 GPU (15 GB VRAM)  
**Final checkpoint:** `sana_psl_medical_finetuned.pt` (~2.1 GB)

---

## 7. Training Results

### 7.1 Phase 1 — ASL Foundation (How2Sign)

| Metric | Value |
|---|---|
| Dataset | How2Sign (31,047 clips) |
| Initial Val Loss | 4.2730 |
| Final Val Loss | 3.4176 |
| Perplexity Reduction | >21% (38.8 → 30.5) |

### 7.2 Phase 2 — PSL Domain Adaptation

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 16.1731 | 4.6220 |
| 2 | 4.2492 | 2.4868 |
| 3 | 2.7309 | 1.4493 |
| 4 | 1.9230 | 0.9169 |
| 5 | 1.5104 | **0.7310** |

**84% validation loss reduction** in 5 epochs on PSL isolated words.

### 7.3 Phase 3 — Medical Fine-Tuning (15-Class Dataset, 30 Epochs)

| Epoch | Loss | Epoch | Loss |
|---|---|---|---|
| 1 | 4.3999 | 16 | 0.1557 |
| 2 | 1.3194 | 17 | 0.1303 |
| 3 | 0.5372 | 18 | 0.1203 |
| 4 | 0.3681 | 19 | 0.0789 |
| 5 | 0.6342 | 20 | 0.0611 |
| 6 | 0.5515 | 21 | 0.2447 |
| 7 | 0.4312 | 22 | 0.1149 |
| 8 | 0.3781 | 23 | 0.0795 |
| 9 | 0.3212 | 24 | 0.0653 |
| 10 | 0.2538 | 25 | 0.0490 |
| 11 | 0.2132 | 26 | 0.0492 |
| 12 | 0.1896 | 27 | 0.0380 |
| 13 | 0.2001 | 28 | 0.0514 |
| 14 | 0.1417 | 29 | 0.0519 |
| 15 | 0.1954 | 30 | **0.0315** |

Loss descended from **4.40 → 0.0315** — a **99.3% reduction** over 30 epochs.  
Final weights saved as `sana_psl_medical_finetuned.pt`.

---

## 8. Live Inference Performance

### 8.1 Key Numbers

| Metric | Value |
|---|---|
| MediaPipe extraction rate | ~17.2 FPS |
| Model inference latency | ~63 ms |
| SIMPACT latency target | 100 ms |
| Margin vs. target | 1.6× faster than required |
| Dataset validation accuracy | 15/15 classes correct (100%) |

### 8.2 Inference Flow

```
User presses SPACE → recording starts
  │
  ▼
MediaPipe extracts 208-dim vector per frame (live)
  │
  ▼
User presses SPACE → recording stops
  │
  ▼
resample_sequence(raw_frames, target=60)
  │
  ▼
pad_to_max_seq_len(resampled, max=100, dim=208)
  │
  ▼
model.mt5.generate(num_beams=4, num_return_sequences=2)
  │
  ▼
Confidence = exp(score₁) / (exp(score₁) + exp(score₂))
  │
  ├─ ≥ 85%  →  Display English + Urdu text
  └─ < 85%  →  "Human Interpreter Required"
```

### 8.3 Confusable Sign Disambiguation

Some PSL signs have very similar hand shapes. The system handles these explicitly:

| Confusable Pair | Disambiguation Method |
|---|---|
| Yes / No | Wrist velocity analysis — only fires when both appear in top-2 AND margin < 25% |
| Mujhay dard kam hai / tez hai | Beam score margin — cleaner separation via beam confidence |

---

## 9. Avatar System — SANA

### 9.1 Avatar Properties

| Property | Value |
|---|---|
| Software | VRoid Studio 2.14 |
| Format | VRM 0.0 (maximum compatibility) |
| Polygons | ~33,343 |
| Materials | 14 |
| Character | Nurse "SANA" |
| Creator | Reyhan |
| Renderer | Three.js WebGL (runs in browser, no install) |

### 9.2 Animation Pipeline — Two Routes

**Route 1 — Custom MediaPipe to FBX**
```
Reference sign video (.MOV)
  └─→ step1_extract_pose.py   (MediaPipe → pose_json/)
      └─→ step2_make_fbx.py   (headless Blender bakes keyframes → .fbx)
          └─→ avatar_player.html  (Three.js renders on Y Bot skeleton)
```

**Route 2 — DeepMotion Cloud (Khizer's route)**
```
Actor films sign with iPhone
  └─→ DeepMotion cloud → .fbx (full finger + body capture)
      └─→ clean_khizer_fbx.py   (strips leg jitter curves)
          └─→ avatar_player.html (Three.js renders on Y Bot skeleton)
```

---

## 10. Safety Design

| Safety Feature | How It Works |
|---|---|
| **Confidence gate** | < 85% confidence → output suppressed, clinician sees "Human Interpreter Required" |
| **Collapse diagnostic** | Press `d` — feeds real input, all-zeros, and random noise to model; verifies it responds differently to each (not a fixed fallback) |
| **Beam decoding** | 4-beam generation produces true probability-weighted hypotheses; not a lookup table |
| **Manual override** | Doctor can manually select or type a phrase to trigger avatar animation |
| **Mirror correction** | Default ON — prevents left/right inversion from front-facing webcam |

---

## 11. Success Metrics

| Metric | Target | Achieved |
|---|---|---|
| Phrase classification accuracy | ≥ 90% | ✅ 100% (dataset validation) |
| Inference latency | ≤ 100 ms | ✅ ~63 ms |
| Bilingual output (English + Urdu) | Required | ✅ mT5-small native Urdu |
| Demo stability | No crashes | ✅ |
| Confidence safety gate | Required | ✅ < 85% suppressed |

---

## 12. Team and Roles

| Name | Role | Contribution |
|---|---|---|
| **Rehan** | ML Lead / Avatar | Model architecture, all training phases, webcam inference pipeline, Three.js avatar player, FBX animation pipeline |
| **Khizer** | Motion Capture / Signer | PSL phrase recording, medical dataset creation, DeepMotion FBX motion capture, live system testing |
| **Reyhan** | Avatar Design | Designed and created the SANA nurse avatar in VRoid Studio 2.14 |

---

## 13. Future Roadmap

| Priority | Goal | When |
|---|---|---|
| High | Expand vocabulary from 15 → 500 medical phrases | Q4 2026 |
| High | Clinical validation study with deaf community | 2027 |
| Medium | Multi-signer generalization (recruit more signers) | 2027 |
| Medium | Mobile-first redesign (tablet at hospital bedside) | 2027 |
| Long-term | Integration with SANA AI HIMS (Hospital Management System) | 2028 |
| Long-term | Generative avatar motion (replace pre-baked animations) | 2028 |

---

*SANA Sign — Giving deaf patients a voice in Pakistan's hospitals.*
