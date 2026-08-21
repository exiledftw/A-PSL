# Project A-PSL: Compute-Efficient Medical Sign Language Translation

## 1. Abstract
Project A-PSL aims to bridge the communication gap between healthcare providers and the Deaf community by developing a two-way Pakistani Sign Language (PSL) translation system. Given the extreme data scarcity of continuous PSL and strict hardware constraints (training on 15GB VRAM T4 GPUs via Kaggle/Colab), traditional raw-video AI approaches are unfeasible. This project relies on a highly optimized, compute-efficient pipeline: leveraging MediaPipe 2D keypoints, Cross-Modal Transformers, Transfer Learning from American Sign Language (ASL), and Low-Rank Adaptation (LoRA) on multilingual language models (mT5). 

This document serves as the living research and architectural whitepaper, tracking all technical decisions, verified methodologies, and current progress.

---

## 2. Core Technical Decisions & Justifications

### 2.1 Why MediaPipe Keypoints Instead of Raw Video?
*   **The Constraint:** Processing continuous raw video requires massive storage (terabytes) and exorbitant GPU VRAM to compute 3D convolutional features (like I3D). 
*   **The Solution:** We extract 2D skeletal coordinates (Face, Hands, Pose) using MediaPipe, reducing a 50MB video into a lightweight JSON file containing coordinates (`Frames × 75 Landmarks × 3 coordinates`).
*   **The Justification:** This strips away irrelevant background noise and lighting conditions, directly isolating human movement. It reduces our dataset size from hundreds of gigabytes down to a fraction that fits effortlessly within Kaggle's working memory, allowing a standard T4 GPU to process sequences without Out-Of-Memory (OOM) crashes.

### 2.2 Why Transfer Learning from ASL to PSL?
*   **The Constraint:** There is no public, large-scale, continuous PSL video dataset. Publicly available PSL data is strictly limited to isolated alphabets or individual words.
*   **The Solution:** We pre-train our Visual Encoder on an existing massive dataset of 100,000 American Sign Language clips (YouTube-ASL).
*   **The Justification:** We are not teaching the model ASL grammar to output ASL; we are teaching the model's visual layers the "physics" of human signing (how hands move, how facial expressions change). Once the visual layers understand how to track spatial-temporal gestures, we freeze the foundation and perform "Domain Adaptation" on our smaller PSL dataset, simply re-mapping the physical movements to new meanings.

### 2.3 Why mT5 and LoRA?
*   **The Constraint:** Training a full Large Language Model (LLM) on a 15GB VRAM GPU is impossible. Furthermore, PSL translation output must support both English and Urdu text.
*   **The Solution:** We use **mT5** (Multilingual Text-to-Text Transfer Transformer), which natively understands English and Urdu. We freeze the entire mT5 model to save VRAM and only train lightweight **LoRA (Low-Rank Adaptation)** matrices injected into its attention layers.
*   **The Justification:** This allows us to "free-ride" on the grammar engine of a massive LLM without bearing the computational cost of training it, focusing 100% of our compute budget purely on the translation bridging layer.

---

## 3. The Minimum Viable Product (MVP) Blueprint
To deliver a working prototype for contests and demonstrations without waiting for the full production model, we employ a tightly constrained, deterministic architecture.

### 3.1 Patient to Doctor (PSL Gestures → Text)
*   **Methodology:** Sequence Classification.
*   **Implementation:** We define a strict vocabulary of 30-50 emergency medical phrases (e.g., "I have a headache," "Where is the pain?"). We record custom videos of these phrases, extract keypoints, and train a lightweight LSTM or Spatial-Transformer classifier.
*   **Justification:** Full autoregressive translation requires massive data. Classification over a fixed vocabulary guarantees >90% accuracy, trains in minutes, and runs flawlessly in real-time on a standard webcam for a live demo.

### 3.2 Doctor to Patient (Voice → Avatar Gestures)
*   **Methodology:** Trigger-Based Avatar Animation.
*   **Implementation:** We utilize OpenAI Whisper (or Google Cloud STT) for real-time Voice-to-Text transcription. The text is mapped via NLP similarity to our 50 predefined phrases. The matched phrase ID triggers a pre-animated 3D avatar sequence (built in Unity/Web Canvas).
*   **Justification:** Real-time generative skeletal AI suffers from severe "uncanny valley" effects and high latency. Triggering pre-rendered, high-quality animations ensures immediate, professional, and socially acceptable feedback for the patient.

---

## 4. Current Progress & Verified Milestones

### ✅ Completed & Verified
- **Architectural Blueprint:** The complete pipeline (Keypoints → Visual Encoder → LoRA mT5) has been documented and verified to theoretically fit within Kaggle T4 parameters.
- **Dataset Storage Hack:** Successfully verified a pipeline to bypass Kaggle's 20GB local storage limit by mounting the massive LINDAT YouTube-ASL zip files directly via the "Remote Files" UI.
- **Checklist Framework:** Established a 6-phase tracker (`checklist.md`) mapping the entire pipeline from raw data to the live MVP demo.

### 🔴 Immediate Next Steps (Phase 1 Execution)
1.  Initialize the public Kaggle dataset by directly linking the 10 LINDAT URL zip files.
2.  Download the official Google Research TSV captions.
3.  Execute a Python script to join the JSON keypoint files to the TSV captions using `video_id` and timestamp composite keys.
4.  Downsample the dataset to a balanced 100,000 clips to prepare for Phase 3 ASL Pre-training.

---
*Note: This document is strictly bounded by what is technically feasible under current hardware constraints. No assumptions are made regarding the existence of continuous PSL datasets; all PSL data required for Phase 3 and the MVP will be custom-recorded under constrained protocols.*