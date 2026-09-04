# Phases Document
# SANA Sign — Project Phases & Timeline

> **Project:** SANA Sign (A-PSL)
> **Event:** SIMPACT 2026 — September 17, 2026
> **This document covers:** What was done in each phase, what is active now, and what comes next.

---

## Phase Overview

```
Phase 1  ──→  Phase 2  ──→  Phase 3  ──→  Phase 4  ──→  Phase 5  ──→  Phase 6 (NOW)  ──→  Phase 7
ASL         PSL Domain    Medical       Live Webcam   Avatar       SIMPACT MVP       Post-Contest
Foundation  Adapt         Fine-Tune     Inference     System       Integration       Scale
[DONE]      [DONE]        [DONE]        [DONE]        [DONE]       [ACTIVE]          [FUTURE]
```

---

## Phase 1 — ASL Foundation Pre-Training
**Status:** COMPLETE
**Date:** August 2026 (early sessions)
**Owner:** Rehan

### What Was Done
- Trained Visual Encoder + mT5-small Decoder on How2Sign (31k ASL clips)
- Achieved Val Loss: 3.6581 (baseline)
- Cross-attention fine-tuning on How2Sign: Val Loss improved to 3.4176 (21% perplexity drop)
- Dataset strategy: YouTube-ASL keypoints (390k clips) on LINDAT → Kaggle remote import
- Implemented Conv1D Temporal Tokenizer (stride=2) → 300 noisy frames → 75 gesture tokens (16x attention noise cut)

### Key Files
- A-PSL_Phase1-2-3.ipynb
- plans/strategy_1-YT-ASL-H2S.md
- dataset_strategy.md

### Architecture Decision
Used Visual Encoder + Projection Layer + frozen mT5 Decoder, then LoRA adapters for fine-tuning.
This avoids the VRAM limits of training a full LLM on a T4 (15GB).

---

## Phase 2 — PSL Domain Adaptation
**Status:** COMPLETE
**Date:** August 2026 (mid sessions)
**Owner:** Rehan + Khizer

### What Was Done
- Adapted pre-trained ASL model to Pakistani Sign Language (71-word PSL dataset from mohib123456 Kaggle)
- Few-shot adaptation: 5 epochs, Val Loss dropped from 4.62 to 0.73 (84% reduction)
- Live translation validation on held-out PSL samples:
  - "you" → "you" (100% exact match, 42.9ms)
  - "deer" → "deer" (100% exact match, 61.3ms)
  - "policecar" → "policecar" (100% exact match, 64.7ms)
  - "bayan haath" → "haath" (Urdu semantic match, 85.2ms)
- Proved mT5 can decode visual sign features directly to Urdu script

### Key Files
- SANA_PSL_FewShot_Train.ipynb
- SANA_Optuna_Sweep.ipynb
- plans/strategy_2-PSL-Medical.md

---

## Phase 3 — Medical Domain Fine-Tuning
**Status:** COMPLETE
**Date:** Late August 2026
**Owner:** Khizer (recording) + Rehan (training)

### What Was Done
- Recorded ~40 PSL medical phrase videos on iPhone (Khizer as signer)
- Extracted 208-dim MediaPipe keypoints from videos (keypoints.ipynb)
- Fine-tuned model on medical vocabulary: 4 phrase classes initially
  - Assalam o Alaikum
  - There has been an accident
  - Is blood pressure high or low
  - Test are cheap here
- Saved: sana_psl_medical_finetuned.pt (~2.1GB full checkpoint)
- Dataset guidelines documented: DATASET_RECORDING_GUIDELINES.md

### Key Files
- SANA_PSL_Dataset_Probe.ipynb
- keypoints.ipynb
- medical_dataset/ folder

---

## Phase 4 — Live Webcam Inference Pipeline
**Status:** COMPLETE
**Date:** September 1-2, 2026
**Owner:** Rehan + Khizer (testing)

### What Was Done
- Discovered critical pipeline misalignment: live script was NOT matching keypoints.ipynb
- Root cause: live script was zero-padding directly to 100 frames, skipping the 60-frame resample step
- Fix: two-step process (resample → 60, then pad → 100) matches 1D Conv temporal learning
- Rewrote webcam_inference.py from scratch with:
  - MediaPipe Tasks API (HandLandmarker + PoseLandmarker)
  - Exact 208-dim vector construction [66 pose, 42 L-hand, 42 R-hand, 58 zeros]
  - Exponential smoothing alpha=0.75
  - Mirror correction toggle (m key)
  - Collapse diagnostic (d key)
  - Confidence gating: < 85% → suppress

### Key Results
- All 4 phrase classes correctly predicted on live webcam
- Average latency: ~63ms (30x faster than SIMPACT requirement)
- Collapse diagnostic confirmed model is reactive (not fixed fallback)

### Key Files
- webcam_inference.py
- HANDOFF.md (full technical breakdown)
- memory/khizer_2026-09-02_memory.md

---

## Phase 5 — Avatar System (3D Sign Language Display)
**Status:** COMPLETE
**Date:** September 3-4, 2026
**Owner:** Rehan (avatar/code), Khizer (motion capture), Reyhan (avatar design)

### What Was Done

#### 5A — Custom FBX Pipeline (MediaPipe → FBX)
- step1_extract_pose.py: extracts 3D landmarks from MOV video → pose_json/
- step2_make_fbx.py: headless Blender script → bakes keyframes → medical_dataset/*.fbx
- Successfully processed "how are you doing.MOV" → medical_dataset/how_are_you_doing.fbx (154 frames @ 30fps)

#### 5B — Three.js Avatar Player
- avatar_player.html: loads Y Bot.fbx skeleton, plays FBX/BVH animations in browser
- Fixed fixDuplicateSkeleton() bug (dual skeleton binding issue in Three.js)
- Fixed mixamorig: prefix stripping for FBX track binding
- Fixed BVH retargeting: bvhQuat × restQuat formula
- Added smoothQuaternionTrack() with spike rejection and 5-tap Gaussian kernel
- Hip root locking: X/Z locked, Y dampened 85%
- Torso stabilization: Spine/Neck/Head tracks damped 65%
- Left arm natural pose: quaternions from paani.fbx (not frozen in T-pose)
- FREEZE_BONES reduced to lower body only (fixed two-handed sign support)

#### 5C — Khizer's DeepMotion BVH/FBX
- Khizer filmed signs wearing motion capture suit, processed via DeepMotion cloud
- Received BVH and FBX with full finger capture (400 curves in how_do_you_feel.fbx)
- Cleaned jitter: stripped 80 leg curves from Khizer FBX → how_do_you_feel_clean.fbx
- BVH spike analysis: 53.4 degree spike at frame 39, 67.9 degree at frame 167 (mitigated by smoothing)

#### 5D — SANA Avatar (VRoid)
- Created nurse avatar "SANA" in VRoid Studio 2.14
- Creator: Reyhan, Version 1.0.0, Copyright: 2026 Bunby
- Stats: ~33,343 polygons, 14 materials
- Exported as VRM 0.0 for maximum compatibility
- Tested in XR Animator and Warudo

### Key Files
- avatar_player.html (main player)
- avatar_bvh_player.html
- step1_extract_pose.py
- step2_make_fbx.py
- medical_dataset/
- bvh_files/ (I_was_born_deaf, do_you_feel_dizzy, how_do_you_feel, keep_your_feet_moisturized)

### Decisions Made
- Rejected Rokoko ("sucks"), Kalidoface (too jittery), Freemocap (overhead)
- Chose XR Animator for browser-based live capture
- Chose Warudo for smooth desktop live capture
- Abandoned FBX file cleaning in Blender ("waste of time")
- Chose VRM 0.0 over VRM 1.0 for broader compatibility

---

## Phase 6 — SIMPACT MVP Integration (ACTIVE NOW)
**Status:** IN PROGRESS
**Date:** September 4-17, 2026
**Owner:** Rehan (integration)

### What Needs to Be Done
- [ ] Connect Pipeline A (webcam inference) output → display panel
- [ ] Connect Pipeline B (Whisper STT → phrase match → avatar animation trigger)
- [ ] Add confidence display UI with red warning for < 85%
- [ ] Build phrase selection fallback UI (doctor manually picks phrase)
- [ ] Final avatar polish: reduce head softness, fix salute distance in VRoid
- [ ] Lock avatar position (no drift, gestures only)
- [ ] Add all 40 medical phrases as FBX animations or BVH
- [ ] Demo script: 10-minute walkthrough with 5 example interactions
- [ ] Create consent form and clinical validation plan document

### Key Milestone
**September 17, 2026** — SIMPACT 2026 showcase at CIME Karachi

---

## Phase 7 — Post-Contest Scale (FUTURE)
**Status:** PLANNED
**Owner:** TBD

### Roadmap Items
- Expand vocabulary from 40 → 500 medical phrases
- Recruit additional PSL signers for dataset diversity
- Retrain with beam search decoding (reduce collapse risk)
- Move from constrained classification to open T5 generation
- Clinical validation study with deaf community
- Mobile-first redesign (tablet at hospital bedside)
- Integration with SANA AI HIMS (Hospital Information Management System)
- Generative avatar motion (replace pre-baked FBX with realtime motion synthesis)

---

## Timeline Summary

| Phase | Description | Status | Date |
|---|---|---|---|
| 1 | ASL Foundation Pre-training (How2Sign + YT-ASL) | DONE | Aug 2026 |
| 2 | PSL Domain Adaptation (71-word PSL dataset) | DONE | Aug 2026 |
| 3 | Medical Fine-tuning (40 videos, 4 phrases) | DONE | Late Aug 2026 |
| 4 | Live Webcam Inference Pipeline | DONE | Sep 1-2, 2026 |
| 5 | Avatar System (FBX pipeline + SANA VRM) | DONE | Sep 3-4, 2026 |
| 6 | SIMPACT MVP Integration | ACTIVE | Sep 4-17, 2026 |
| 7 | Post-Contest Scale | FUTURE | After Sep 17 |
