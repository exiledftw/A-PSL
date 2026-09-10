# SANA Sign — Team Progress Report
### Internal Record | Rehan, Khizer, Reyhan

> **Project:** SANA Sign (A-PSL)
> **Event:** SIMPACT 2026 — CIME Karachi, September 17, 2026
> **Last Updated:** September 10, 2026

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Full Timeline — Session by Session](#2-full-timeline--session-by-session)
3. [The Three Training Phases](#3-the-three-training-phases)
4. [The Dataset We Made](#4-the-dataset-we-made)
5. [The Live Webcam System](#5-the-live-webcam-system)
6. [The SANA Avatar](#6-the-sana-avatar)
7. [What Is Working Right Now](#7-what-is-working-right-now)
8. [Current File Locations](#8-current-file-locations)
9. [Lessons Learned](#9-lessons-learned)
10. [Demo Day Plan](#10-demo-day-plan)

---

## 1. What We Built

SANA Sign is an AI-powered sign language translator for Pakistani hospital use. There are two main parts:

**Part 1 — Patient talks to doctor via PSL gestures:**
Patient does a sign in front of a webcam → AI recognises it → screen shows the English and Urdu meaning to the doctor.

**Part 2 — Doctor talks back through a 3D avatar:**
Doctor speaks → system hears → our 3D nurse avatar "SANA" performs the matching PSL sign so the patient understands.

---

## 2. Full Timeline — Session by Session

### August 20 — Project Started
- Set up GitHub repo (`exiledftw/A-PSL`) and shared it between Khizer and Rehan
- Decided on the model plan: Visual Encoder + frozen mT5 decoder, trained on keypoints only (not raw video)
- Dataset strategy: start with How2Sign (31k ASL clips), then adapt to PSL
- Set up Notion project board with dataset strategy and fairness/bias notes

### August 21 — Architecture Finalised
- Decided on the 208-dim feature vector: 33 pose + 21 left hand + 21 right hand + 58 face (zeros)
- Chose MediaPipe Tasks API over MediaPipe Holistic (newer, faster)
- Implemented the Conv1D Temporal Tokenizer concept — compresses 300 noisy frames into 75 gesture tokens

### August 24 — Hyperparameter Tuning
- Ran Optuna sweep on Kaggle using 10,000 clip subset to find best settings
- Found final hyperparameters:
  - Learning Rate: `2.00e-4`
  - Weight Decay: `6.1357e-4`
  - Dropout: `0.108`
  - 2 encoder layers, FFN=1024
  - Batch size: 4, Gradient accumulation: 2 steps

### August 25 — Phase 1 Training Started (Rehan)
- Began full training on How2Sign (31,047 ASL clips) on Kaggle T4
- Set up Weights & Biases live tracking
- Epoch 1 val loss: **4.2730** (already better than the baseline from Optuna at 7.93)
- Average epoch duration: ~18.2 minutes

### August 27 — Phase 1 Complete
- Final val loss: **3.4176**
- Perplexity dropped 21% (38.8 → 30.5)
- Identified "language model prior trap" — model was collapsing to generic phrases
- Fix: added Conv1D Temporal Tokenizer to sharpen attention focus on hand motion

### August 28 — PSL Domain Adaptation (Phase 2)
- Few-shot fine-tuned on 71-word PSL dataset from Kaggle (mohib123456)
- Results in 5 epochs:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 16.17 | 4.62 |
| 2 | 4.25 | 2.49 |
| 3 | 2.73 | 1.45 |
| 4 | 1.92 | 0.92 |
| 5 | 1.51 | **0.73** |

- Validated live on PSL samples:
  - "you" → "you" ✅ (100% exact, 42.9ms)
  - "deer" → "deer" ✅ (100% exact, 61.3ms)
  - "policecar" → "policecar" ✅ (100% exact, 64.7ms)
  - "بایاں ہاتھ" → "ہاتھ" ✅ (Urdu semantic match, 85.2ms)
- Average latency: **~63ms** — 30× faster than SIMPACT's 100ms requirement

### August 31 — Medical Phase Planned
- Retired YouTube-ASL approach (How2Sign foundation is stable enough)
- Decided to self-record 40 medical triage phrases (Khizer + Rehan as signers)
- Planned 10 videos per phrase × 6× augmentation = 60 training samples per class

### September 1 — Keypoint Extraction Pipeline
- Built `keypoints.ipynb` for Google Colab
- Features:
  - Loads videos from Google Drive
  - Selfie mirror auto-correction
  - 60-frame temporal resampling
  - 6× data augmentation
  - Saves `.npy` keypoint files
  - Zips and backs up to Google Drive

### September 2 — Live Webcam Breakthrough
- Discovered the pipeline mismatch that was causing wrong predictions
- **Root cause:** Live script was zero-padding directly to 100 frames, skipping the 60-frame resample step that training used
- **Fix:** Two-step `resample → 60 frames → pad to 100` matching training exactly
- Rewrote `webcam_inference.py` from scratch with this correct pipeline
- All 4 test classes predicted correctly

### September 3-4 — Avatar System (Phase 5)
- Built `step1_extract_pose.py` and `step2_make_fbx.py` (MediaPipe → FBX pipeline)
- Built `avatar_player.html` in Three.js
- Fixed major Three.js bugs: dual skeleton, bone prefix, BVH retargeting formula
- Khizer's DeepMotion FBX files cleaned and integrated
- SANA avatar created by Reyhan in VRoid Studio 2.14 (VRM 0.0)

### September 7 — NLP Audio Trigger System
- Built `xr_pipeline/audio_listener.py` for speech recognition with fuzzy matching
- Built `xr_pipeline/stream_controller.py` for UDP-triggered video streaming

### September 8-9 — New Dataset Recording
- Khizer and Rehan recorded new medical dataset
- 15 phrase classes, 60 keypoint files each
- Dataset saved to `E:\sign-language\dataset`

### September 10 — Final Medical Fine-Tuning Complete
- Ran 30-epoch medical fine-tuning on new 15-class dataset
- Final loss: **0.0315** (started at 4.40)
- Model verified: 15/15 classes predicted correctly on dataset samples
- Saved as `sana_psl_medical_finetuned.pt` (~2.1 GB)
- HUD display updated: Urdu text rendering, confidence %, runner-up phrase shown

---

## 3. The Three Training Phases

### Phase 1 — How2Sign ASL Foundation

| What | Value |
|---|---|
| Dataset | How2Sign (31,047 ASL clips from YouTube) |
| Epochs | 15 |
| Starting val loss | 4.2730 (Epoch 1) |
| Final val loss | **3.4176** |
| Perplexity improvement | 21% reduction |
| Hardware | Kaggle T4 (15 GB VRAM) |

### Phase 2 — PSL Domain Adaptation

| What | Value |
|---|---|
| Dataset | 71-word PSL dataset (mohib123456/Kaggle) |
| Epochs | 5 |
| Starting val loss | 4.6220 |
| Final val loss | **0.7310** |
| Improvement | 84% loss reduction |

### Phase 3 — Medical Fine-Tuning (Final Model)

30 epochs on our own 15-class medical dataset:

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

Started at 4.40, ended at 0.0315 — **99.3% loss reduction.**

---

## 4. The Dataset We Made

### What It Is
We recorded our own PSL medical phrases because there is no public PSL medical dataset. The data lives at `E:\sign-language\dataset`.

### The 15 Phrases

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

### Recording Details
- **Signers:** Khizer + Rehan
- **60 keypoint files per class** (no videos saved, just keypoints)
- **Shape:** each file is `(60, 208)` — 60 frames × 208 features
- **6× augmentation** was applied to multiply training data

---

## 5. The Live Webcam System

### How to Run

```bash
python webcam_inference.py
```

### Controls

| Key | What It Does |
|---|---|
| SPACE | Start/stop recording |
| m | Toggle mirror correction |
| d | Run diagnostic test |
| q | Quit |

### What You See

- Live webcam feed with skeleton overlay
- After recording: prediction result shown in HUD
- Top prediction with confidence %
- Runner-up prediction
- Urdu translation rendered on screen
- If confidence < 85%: warning shown instead

### Performance

| Metric | Value |
|---|---|
| MediaPipe speed | ~17.2 FPS |
| Model inference | ~63 ms |
| SIMPACT target | 100 ms |
| Accuracy on dataset | 15/15 classes (100%) |

---

## 6. The SANA Avatar

### The Avatar
- **Name:** SANA
- **Type:** 3D nurse character
- **Created by:** Reyhan in VRoid Studio 2.14
- **Format:** VRM 0.0 (~33k polygons, 14 materials)

### How Animations Work
Khizer recorded PSL signs on his iPhone and used DeepMotion (cloud service) to convert them to FBX animation files. These were cleaned and loaded into our Three.js browser player.

Rehan also built a custom pipeline:
- Record reference video (.MOV)
- Run `step1_extract_pose.py` → extracts 3D pose to JSON
- Run `step2_make_fbx.py` → bakes into FBX using Blender

The avatar player runs in Chrome, no installation needed:
```bash
python -m http.server 8000
# Open: http://localhost:8000/avatar_player.html
```

---

## 7. What Is Working Right Now

| Feature | Status |
|---|---|
| Live PSL gesture recognition (15 classes) | ✅ Working |
| English + Urdu output on screen | ✅ Working |
| Confidence gate (< 85% suppressed) | ✅ Working |
| Mirror correction toggle | ✅ Working |
| Diagnostic mode (d key) | ✅ Working |
| SANA avatar in browser | ✅ Working |
| DeepMotion FBX animations | ✅ Working |
| NLP audio trigger (Pipeline B) | ✅ Built |

---

## 8. Current File Locations

| Item | Location |
|---|---|
| Main inference script | `E:\sign-language\Beta\webcam_inference.py` |
| Final model | `E:\sign-language\Beta\sana_psl_medical_finetuned.pt` |
| Dataset (keypoints) | `E:\sign-language\dataset\` |
| Training notebook backup | `E:\sign-language\backup\notebook399144a5d3.ipynb` |
| Avatar player | `E:\sign-language\Beta\avatar_player.html` |
| All documentation | `E:\sign-language\Beta\project docs\` |
| Session memory logs | `E:\sign-language\Beta\memory\` |

---

## 9. Lessons Learned

**The 60-frame resample rule (September 2)**
The biggest issue we hit was that the live script was padding directly to 100 frames instead of first resampling to 60. The Conv1D layers learned temporal patterns at 60 frames. This one mismatch caused completely wrong predictions across all signs. Always: resample to 60 first, then pad to 100.

**Beam generation vs. candidate scoring**
We tried computing the model's loss on each candidate phrase and picking the lowest. This failed because longer phrases naturally have lower average token loss. Switch to beam generation — it's faster and actually correct.

**Mirror correction is not optional**
The training data was recorded with a selfie camera where MediaPipe's "Left" label actually means the signer's right hand. If mirror correction is off during demo, all sign classifications will be wrong.

**DeepMotion vs custom FBX**
DeepMotion gives much cleaner finger captures than our custom MediaPipe → Blender pipeline. For demo quality, use DeepMotion files wherever possible.

---

## 10. Demo Day Plan

**Date:** September 17, 2026 — SIMPACT 2026, CIME Karachi

### Setup
1. Open `webcam_inference.py` (keep it running in background)
2. Open `avatar_player.html` in Chrome at `http://localhost:8000/avatar_player.html`
3. Position laptop so webcam faces the signer clearly
4. Verify mirror correction is ON (`m` key indicator in HUD)

### Demo Flow (10 minutes)

1. **Introduction (2 min):** Explain the problem — deaf patients in hospitals
2. **Pipeline A demo (4 min):**
   - Signer performs "Assalam o Alaikum" → system shows greeting in English + Urdu
   - Signer performs "Mujhay bukhar hai" → system shows "I have fever"
   - Signer performs "Ambulance ko call karro" → system shows emergency phrase
   - Show confidence % on screen
3. **Pipeline B demo (3 min):**
   - Doctor speaks "does the patient feel dizzy" → avatar performs PSL sign
4. **Safety feature demo (1 min):**
   - Press `d` to show diagnostic — proves model is live, not pre-recorded

### Backup Plan
If webcam doesn't detect hands well at venue lighting:
- Press `m` to toggle mirror if signs come out backwards
- Slow down signing speed — model needs full gesture captured within SPACE presses
- Move to a lighter background area
