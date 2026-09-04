# Product Requirements Document (PRD)
# SANA Sign — Pakistani Sign Language Doctor-Patient Communication System

> **Project:** SANA Sign (A-PSL)
> **Event:** SIMPACT 2026 — CIME Karachi, September 17, 2026
> **Version:** Phase 6 MVP
> **Team:** Rehan (ML / Avatar), Khizer (Motion Capture / FBX), Reyhan (Avatar Design)
> **Status:** 🟡 Active Development

---

## 1. Problem Statement

Pakistan has **approximately 1 million deaf and hard-of-hearing individuals**. When a deaf patient visits a hospital, communication between doctor and patient breaks down entirely. There are:

- **No certified PSL interpreters** available at most hospitals
- No assistive technology for sign language communication in clinical settings
- **Life-threatening mistranslations** during emergency triage
- Total reliance on handwritten notes — slow, error-prone, and unusable during physical examinations

### The Gap

```
  Deaf Patient ──[PSL]──→ ??? ──→ Doctor (Urdu/English Speaker)
  Doctor (Urdu/English) ──→ ??? ──→ Deaf Patient
```

There is nothing in the middle. SANA Sign fills this gap.

---

## 2. Vision

> **"A deaf patient should be able to walk into any hospital in Pakistan and communicate with a doctor — without an interpreter."**

SANA Sign is a bilingual, AI-powered, two-way communication bridge that:
1. Translates **Pakistani Sign Language (PSL) gestures → Urdu/English text** for the doctor
2. Converts **doctor's spoken words → 3D Avatar PSL animations** for the deaf patient

---

## 3. Users & Personas

### Persona A: The Deaf Patient (Primary)
| Attribute | Detail |
|---|---|
| **Language** | Pakistani Sign Language (PSL) |
| **Literacy** | May have limited Urdu/English literacy |
| **Tech Comfort** | Minimal — uses simple touchscreen |
| **Critical Need** | To express symptoms accurately and understand diagnosis |
| **Frustration** | Writing on paper is slow; doctors do not know PSL |

### Persona B: The Doctor / Clinician (Secondary)
| Attribute | Detail |
|---|---|
| **Language** | Urdu and/or English |
| **Tech Comfort** | Moderate — comfortable with digital tools |
| **Critical Need** | To receive the patient's complaint in readable text |
| **Frustration** | Guesswork, sign-note mismatch, wasted consultation time |

### Persona C: Hospital Admin / SIMPACT Evaluator (Tertiary)
| Attribute | Detail |
|---|---|
| **Goal** | Clinical viability, scalability, safety compliance |
| **Critical Need** | Confidence threshold, error handling, audit trail |

---

## 4. Core Features (MVP Scope)

### 4.1 Patient → Doctor Track (Gesture-to-Text)

| Feature | Requirement |
|---|---|
| **Webcam Input** | Real-time PSL sign capture via standard laptop camera |
| **Gesture Recognition** | Classify 40+ pre-defined medical PSL phrases |
| **MediaPipe Extraction** | 208-dim feature vector: 33 pose + 21L hand + 21R hand landmarks (x,y) |
| **Confidence Display** | Show confidence %; suppress if < 85% → show "Human Interpreter Required" |
| **Output** | English AND Urdu translation displayed on screen |
| **Latency** | 100ms inference after capture |

**Supported Phrase Categories:**
- Greetings & Initial Complaint (8 phrases)
- Describing Symptoms (20 phrases)
- Medical History Answers (9 phrases)
- Questions for the Doctor (9 phrases)
- Closing (2 phrases)

### 4.2 Doctor → Patient Track (Text-to-Avatar)

| Feature | Requirement |
|---|---|
| **Voice Input** | Whisper / Google STT captures doctor's speech |
| **Text-to-Gloss** | Fuzzy NLP matching maps sentence to known phrase ID |
| **Avatar Display** | 3D nurse avatar (SANA) performs PSL sign animation |
| **Animation System** | Pre-baked FBX/BVH animations triggered on phrase match |
| **Avatar Name** | SANA — VRoid-based nurse avatar, VRM 0.0 format |
| **Renderer** | Three.js WebGL in browser (no install required) |

### 4.3 Safety & Fallback

| Feature | Requirement |
|---|---|
| **Low Confidence Flag** | Confidence < 85% → suppress output, prompt clinician |
| **Collapse Diagnostic** | Diagnostic mode: feeds 3 contrasting inputs to model to verify no collapse |
| **Fallback Message** | Display "Sign not recognized — please try again" |
| **Override** | Doctor can manually type or select phrase to force avatar animation |

---

## 5. Out of Scope (MVP)

| Not in MVP | Reason |
|---|---|
| Open-vocabulary PSL translation | Requires 1000+ hours training data |
| Real-time generative avatar motion | Too complex; uncanny valley risk |
| Mobile app (iOS/Android) | Web-first for SIMPACT demo |
| Clinical certification | Post-prototype validation study |
| Multi-signer generalization | 1 trained signer (Khizer) for MVP |

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| Phrase classification accuracy (held-out) | 90% or higher |
| Inference latency | 100ms or less |
| Avatar animation clarity (evaluator rating) | 4/5 or higher |
| Demo stability (no crashes during 10-min demo) | 100% |
| SIMPACT reviewer pass on safety design | Pass |

---

## 7. Constraints

- **Hardware:** Standard laptop webcam (no depth camera)
- **Compute:** T4 GPU on Kaggle (15GB VRAM)
- **Storage:** Kaggle 20GB scratch + dataset storage
- **Timeline:** September 17, 2026 (SIMPACT 2026 showcase)
- **Platform:** Web browser (no Unity/Unreal required for demo)
- **Dataset Size:** ~40 training videos (medical fine-tune), 71-word PSL dataset (Kaggle)

---

## 8. Dataset

### Training Data Summary
| Dataset | Role | Size | Source |
|---|---|---|---|
| How2Sign | ASL Foundation Pre-training | ~31k clips | Public |
| YouTube-ASL (keypoints) | Diversity pre-training | 390k JSON clips | LINDAT via Kaggle |
| PSL Medical Custom | Medical fine-tuning | ~40 videos | Recorded by Khizer |
| PSL Isolated Words | PSL domain adaptation | 71 words | mohib123456 Kaggle |

### Recording Protocol
- iPhone 1080p at 30fps, landscape, AE/AF locked
- Plain background, solid dark clothing, short sleeves
- "Return to Zero" rule: arms down before and after each sign
- Transfer via USB or Google Drive (NOT WhatsApp — compression kills quality)
- See DATASET_RECORDING_GUIDELINES.md for full rules

---

## 9. Non-Technical Deliverables (SIMPACT Requirement)

- [ ] PSL Dataset documentation (demographics, recording conditions)
- [ ] Video consent flow for clinical capture
- [ ] Clinical validation study design (post-prototype)
- [ ] Avatar acceptance testing with deaf community feedback
- [ ] Confidence threshold rationale document

---

## 10. Stakeholder Sign-Off

| Role | Name | Status |
|---|---|---|
| ML Lead / Avatar | Rehan | Active |
| Motion Capture | Khizer | Active |
| Avatar Design | Reyhan | SANA Created |
| Event | SIMPACT 2026 | Sep 17, 2026 |
