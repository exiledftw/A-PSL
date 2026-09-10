# EMRChains — SANA AI
### Developer Handoff Document & Technical Architecture Guide
#### For: Engineering & Infrastructure Contributors

> **Project:** EMRChains — SANA AI
> **Live Web Application:** [https://psl-sana.vercel.app/](https://psl-sana.vercel.app/)
> **Backend AI Hosting:** Oracle Cloud Infrastructure
> **Repository:** `exiledftw/A-PSL` (Local: `E:\sign-language\Beta`)
> **Last Updated:** September 10, 2026
> **Authors:** Rehan + Khizer

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository & Cloud Topology](#2-repository--cloud-topology)
3. [Critical Model Contracts](#3-critical-model-contracts)
4. [The AI Translation Model](#4-the-ai-translation-model)
5. [The Patient Dataset (15 Medical Classes)](#5-the-patient-dataset-15-medical-classes)
6. [Doctor-to-Patient Pipeline (13 Doctor Phrases)](#6-doctor-to-patient-pipeline-13-doctor-phrases)
7. [Oracle Cloud AI Backend API](#7-oracle-cloud-ai-backend-api)
8. [Frontend & Authentication (Next.js on Vercel)](#8-frontend--authentication-nextjs-on-vercel)
9. [Local Development & Live Inference](#9-local-development--live-inference)
10. [Environment Setup & Dependencies](#10-environment-setup--dependencies)
11. [Key Scripts Reference](#11-key-scripts-reference)
12. [Glossary](#12-glossary)

---

## 1. Architecture Overview

EMRChains — SANA AI bridges hospital communication across two real-time pipelines:

- **Pipeline A (Patient → Doctor):**
  1. Patient performs a PSL sign on the Patient Dashboard webcam.
  2. The video is pushed to the **Oracle Cloud AI backend**.
  3. MediaPipe Tasks extracts 208-dimensional landmark coordinates frame-by-frame.
  4. Temporal resampler normalizes sequence length to 60 frames, then zero-pads to 100 frames.
  5. The `SANA_PSL_Translator` model executes 4-beam sequence generation.
  6. The API returns the top predicted English and Urdu translations with confidence scores.
  7. Results render live on the Doctor Console.

- **Pipeline B (Doctor → Patient):**
  1. Doctor speaks in English or Urdu on the Doctor Console.
  2. Speech recognition converts audio to text.
  3. NLP router fuzzy-matches the speech to one of **13 doctor phrases**.
  4. A **preview video** of the verified, pre-recorded human PSL sign appears on the Doctor Console.
  5. Doctor verifies the sign matches clinical intent and clicks **Send**.
  6. The pre-recorded `.mp4` video (hosted on the Vercel web app) plays automatically on the Patient Dashboard.

---

## 2. Repository & Cloud Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Architecture                       │
│                                                             │
│   Vercel Deployment (https://psl-sana.vercel.app/)          │
│   ├── Next.js Full-Stack App                                │
│   ├── Pre-recorded PSL Videos (.mp4 library)                │
│   ├── Doctor Console (admin / admin123)                     │
│   └── Patient Dashboard                                     │
│                │                                            │
│                ▼ HTTPS REST API                             │
│   Oracle Cloud AI Backend                                   │
│   ├── FastAPI / Flask Inference Gateway                     │
│   ├── MediaPipe Tasks (PoseLandmarker + HandLandmarker)     │
│   ├── PyTorch SANA_PSL_Translator                           │
│   └── sana_psl_medical_finetuned.pt (~2.1 GB)               │
└─────────────────────────────────────────────────────────────┘
```

### Local Repository Structure

```
E:\sign-language\Beta\
├── webcam_inference.py          ← Core PyTorch inference engine & local live tester
├── record_medical_dataset.py    ← Captures new keypoint training data from webcam
├── text_to_gloss_parser.py      ← NLP matching logic for doctor speech intents
│
├── dataset/                     ← 15 Patient Phrase Classes (60 .npy files each)
│   ├── Assalam_o_alaikum/       ← shape: (60, 208)
│   ├── Yes/
│   ├── No/
│   ├── Mein beemar hu/
│   ├── Mujhay bukhar hai/
│   ├── Mere sarr mein dard hai/
│   ├── Meri aankh surkh hai/
│   ├── Mujhay chakkar aa rhy hein/
│   ├── Mujhay dard kam hai/
│   ├── Mujhay dard tez hai/
│   ├── Ambulance ko call karro/
│   ├── There_has_been_an_accident/
│   ├── is_blood_pressure_high_or_low/
│   ├── Test_are_cheap_here/
│   └── Mujhay kuch dawa khareedni hai/
│
├── memory/                      ← Project context logs & history
├── project docs/                ← Official project documentation
│   ├── SANA_Sign_SIMPACT_Report.md
│   ├── SANA_Sign_Developer_Handoff.md   (THIS FILE)
│   └── SANA_Sign_Team_Report.md
│
└── sana_psl_medical_finetuned.pt  ← Production checkpoint (Flat OrderedDict, ~2.1 GB)
```

---

## 3. Critical Model Contracts

> ⚠️ **CRITICAL:** Any alteration to these dimensions or preprocessing steps without full retraining will corrupt model predictions.

### 3.1 Feature Vector Layout (208 Dimensions per Frame)

```
[  0 :  66 ]  → 33 Pose landmarks × (x, y)          (MediaPipe PoseLandmarker)
[ 66 : 108 ]  → 21 Left Hand landmarks × (x, y)     (MediaPipe HandLandmarker)
[108 : 150 ]  → 21 Right Hand landmarks × (x, y)    (MediaPipe HandLandmarker)
[150 : 208 ]  → Face coordinates: Always 58 Zeros  (SANA standard feature padding)
```

**Handedness Convention:** The training data was recorded via front-facing selfie cameras. MediaPipe detects a front camera's right hand as "Left". The recording pipeline swapped handedness so that the dominant hand maps into the Left Hand slot `[66:108]`. Live inference maintains this contract via `MIRROR_CORRECTION = True`.

### 3.2 Two-Step Temporal Resampling Contract

```
Raw Frames (Variable N)
    │
    ▼  resample_sequence(raw_seq, target_frames=60)
Exact 60 Frames (Linear interpolation across 208 dimensions)
    │
    ▼  pad_to_max_seq_len(resampled, max_len=100, dim=208)
Tensor Shape: [100 × 208] (60 frames of motion + 40 zero-padded frames)
```

The Conv1D temporal kernels were trained on this precise temporal density. Do not pad directly to 100 frames without first resampling to 60.

### 3.3 Checkpoint Format
The checkpoint `sana_psl_medical_finetuned.pt` is a **flat `OrderedDict`** containing layer weights directly:
```python
checkpoint = torch.load("sana_psl_medical_finetuned.pt", map_location=device)
model.load_state_dict(checkpoint)  # Do NOT use checkpoint["model_state_dict"]
```

---

## 4. The AI Translation Model

### 4.1 Architecture Definition (`SANA_PSL_Translator`)

Defined in `webcam_inference.py`:

```
SANA_PSL_Translator
├── visual_encoder: UpgradedSpatialTemporalEncoder
│   ├── tokenizer: TemporalGestureTokenizer
│   │   ├── Conv1d(in_channels=208, out_channels=256, kernel_size=5, stride=2)
│   │   ├── BatchNorm1d(256) + GELU()
│   │   ├── Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
│   │   └── BatchNorm1d(512) + GELU()
│   │   → Compresses [Batch, 100, 208] down to [Batch, 25, 512]
│   │
│   ├── pos_encoder: SinusoidalPositionalEncoding
│   └── transformer_encoder: 2-layer TransformerEncoder (d_model=512, nhead=8, FFN=1024)
│
└── mt5: AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    → Fine-tuned on cross-attention layers only (decoder.block[i].layer[1])
    → Fully frozen self-attention and feed-forward weights
```

### 4.2 Native Beam Search Inference

```python
# Proper sequence generation with beam scoring
outputs = model.mt5.generate(
    inputs_embeds=visual_embeds,
    num_beams=4,
    num_return_sequences=2,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=20,
)

top1_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
top2_text = tokenizer.decode(outputs.sequences[1], skip_special_tokens=True).strip()

# Softmax over top-2 beam scores
s1 = outputs.sequences_scores[0].item()
s2 = outputs.sequences_scores[1].item()
confidence = math.exp(s1) / (math.exp(s1) + math.exp(s2))
```

---

## 5. The Patient Dataset (15 Medical Classes)

Location: `E:\sign-language\dataset\`  
Contains 15 classes (60 `.npy` files per class, shape `(60, 208)`):

1. `Assalam_o_alaikum`
2. `Yes`
3. `No`
4. `Mein beemar hu`
5. `Mujhay bukhar hai`
6. `Mere sarr mein dard hai`
7. `Meri aankh surkh hai`
8. `Mujhay chakkar aa rhy hein`
9. `Mujhay dard kam hai`
10. `Mujhay dard tez hai`
11. `Ambulance ko call karro`
12. `There_has_been_an_accident`
13. `is_blood_pressure_high_or_low`
14. `Test_are_cheap_here`
15. `Mujhay kuch dawa khareedni hai`

*(Note: "Yesterday" is excluded from production inference).*

---

## 6. Doctor-to-Patient Pipeline (13 Doctor Phrases)

### Workflow:
1. **Speech Capture:** Web Speech API or local Whisper model captures spoken English or Urdu from the physician.
2. **Intent Matching:** `text_to_gloss_parser.py` maps spoken input against the 13 physician phrase intents using fuzzy string matching (`thefuzz`).
3. **Clinical Preview:** The corresponding pre-recorded human sign video loads inside the Doctor Console's verification window.
4. **Physician Dispatch:** Upon pressing "Send", an event notifies the Patient Dashboard (via WebSockets or REST polling) to play the matching video.
5. **Video Delivery:** Pre-recorded `.mp4` video files are served directly from the Vercel application public directory.

---

## 7. Oracle Cloud AI Backend API

The backend server hosted on Oracle Cloud exposes endpoints for real-time inference:

### `POST /api/v1/translate-sign`
- **Request:** FormData or JSON payload containing video frames / base64 video chunk.
- **Backend Pipeline:**
  1. MediaPipe Tasks runs landmark detection across frames.
  2. Landmark matrices are formatted into `(N, 208)`.
  3. Resampled to 60 frames and padded to 100 frames.
  4. PyTorch model executes `model.mt5.generate()`.
- **Response Format:**
```json
{
  "status": "success",
  "prediction": "Mujhay bukhar hai",
  "urdu_text": "مجھے بخار ہے",
  "confidence": 0.942,
  "runner_up": "Mein beemar hu",
  "runner_up_confidence": 0.058,
  "inference_time_ms": 61.8,
  "confidence_passed": true
}
```

---

## 8. Frontend & Authentication (Next.js on Vercel)

- **URL:** `https://psl-sana.vercel.app/`
- **Doctor Portal Access:** Requires username `admin` and password `admin123`.
- **Role Isolation:**
  - Authenticated doctors can toggle between the Doctor Console and Patient Dashboard.
  - Patients can only view and interact with the Patient Dashboard.
- **State Management:** Real-time state synchronization handles incoming doctor videos and webcam gesture submission.

---

## 9. Local Development & Live Inference

To run local inference using your computer's webcam:

```bash
cd E:\sign-language\Beta
python webcam_inference.py
```

### Keyboard Controls:
- **`SPACE`**: Start recording sign → press **`SPACE`** again to finish and classify.
- **`m`**: Toggle mirror-hand correction (Default: **ON**).
- **`d`**: Run collapse diagnostic (verifies model responsiveness on real vs zeros vs noise).
- **`q`**: Quit application.

---

## 10. Environment Setup & Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece
pip install mediapipe opencv-python
pip install Pillow arabic-reshaper python-bidi
pip install thefuzz python-Levenshtein
```

---

## 11. Key Scripts Reference

| Script | Purpose |
|---|---|
| `webcam_inference.py` | Production PyTorch inference engine and local live testbed |
| `record_medical_dataset.py` | Webcam keypoint recording tool for expanding phrase datasets |
| `text_to_gloss_parser.py` | NLP fuzzy intent router matching doctor voice to sign video |
| `backup/notebook399144a5d3.ipynb` | Kaggle fine-tuning notebook containing the 30-epoch training run |

---

## 12. Glossary

| Term | Definition |
|---|---|
| **PSL** | Pakistani Sign Language |
| **mT5** | Multilingual Text-to-Text Transfer Transformer (Google) |
| **Conv1D Tokenizer** | 1D Convolutional front-end compressing frame sequences into gesture tokens |
| **MediaPipe Tasks** | Google machine-learning pipeline for real-time hand and pose tracking |
| **Doctor Preview** | Clinical verification gate requiring physician confirmation before sign video dispatch |
| **Vercel** | Hosting platform for the Next.js frontend and pre-recorded PSL video library |
| **Oracle Cloud** | Compute infrastructure hosting the AI model and MediaPipe processing backend |
