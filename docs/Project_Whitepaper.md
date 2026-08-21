# SANA Sign: Compute-Efficient Medical Sign Language Translation

## 1. Abstract
SANA Sign is a module within the SANA AI healthcare platform (HIMS) aimed at bridging the communication gap between healthcare providers and the Deaf community through a two-way Pakistani Sign Language (PSL) translation system. Currently being developed as a prototype for the SIMPACT 2026 showcase (CIME Karachi), the system is built under extreme data scarcity and strict hardware constraints. Traditional raw-video AI approaches are unfeasible. This project relies on a highly optimized, compute-efficient pipeline: leveraging MediaPipe 2D keypoints, Cross-Modal Transformers, Transfer Learning from American Sign Language (ASL), and Low-Rank Adaptation (LoRA) on multilingual language models (mT5). 

This document serves as the living research and architectural whitepaper, tracking all technical decisions, verified methodologies, and current progress.

---

## 2. Core Technical Decisions & Justifications

### 2.1 Why MediaPipe Keypoints Instead of Raw Video?
*   **The Constraint:** Processing continuous raw video requires massive storage (terabytes) and exorbitant GPU VRAM to compute 3D convolutional features (like I3D). 
*   **The Solution:** We extract 2D skeletal coordinates (Face, Hands, Pose) using MediaPipe, reducing a 50MB video into a lightweight JSON file containing coordinates (`Frames A- 75 Landmarks A- 3 coordinates`).
*   **The Justification:** This strips away irrelevant background noise and lighting conditions, directly isolating human movement. It reduces our dataset size from hundreds of gigabytes down to a fraction that fits effortlessly within Kaggle's working memory, allowing a standard T4 GPU to process sequences without Out-Of-Memory (OOM) crashes.

### 2.2 Why Transfer Learning from ASL to PSL?
*   **The Constraint:** There is no public, large-scale, continuous PSL video dataset. Publicly available PSL data is strictly limited to isolated alphabets or individual words.
*   **The Solution:** We pre-train our Visual Encoder on an existing massive dataset of 100,000 American Sign Language clips (YouTube-ASL).
*   **The Justification:** We are not teaching the model ASL grammar to output ASL; we are teaching the model's visual layers the "physics" of human signing (how hands move, how facial expressions change). Once the visual layers understand how to track spatial-temporal gestures, we freeze the foundation and perform "Domain Adaptation" on our smaller PSL dataset, simply re-mapping the physical movements to new meanings.

### 2.3 Why mT5 and LoRA?
*   **The Constraint:** Training a full Large Language Model (LLM) on a 15GB VRAM GPU is impossible. Furthermore, PSL translation output must support both English and Urdu text.
*   **The Solution:** We use **mT5** (Multilingual Text-to-Text Transfer Transformer), which natively understands English and Urdu. We freeze the entire mT5 model to save VRAM and only train lightweight **LoRA (Low-Rank Adaptation)** matrices injected into its attention layers.
*   **The Justification:** This allows us to "free-ride" on the grammar engine of a massive LLM without bearing the computational cost of training it, focusing 100% of our compute budget purely on the translation bridging layer.

### 2.4 The Hybrid Architecture (Pre-Trained Language + From-Scratch Vision)
*   **The Approach:** The model consists of a Language Half (mT5) and a Vision Half (Visual Encoder). We do not train the language part from scratch; we leverage Google's massively pre-trained mT5 model which already possesses flawless English and Urdu grammar. However, the Visual Encoder is trained completely from scratch on the YouTube-ASL dataset.
*   **The Justification:** The original academic researchers (Zelezny et al.) provided the architectural blueprint but withheld the pre-trained weights for the Visual Encoder. Training the vision half from scratch on our end is mathematically required, but it is fast and feasible precisely because the heavy language lifting is already handled by the pre-trained mT5.

### 2.5 Dataset Downsampling & Epoch Checkpointing
*   **The Approach:** The full YouTube-ASL dataset contains ~600,000 clips distributed across 10 massive zip archives. To hit our ~100k clip prototype target, we will define one "Epoch" as sequentially training across exactly 3 zip files (~117,000 clips). The Kaggle training loop will execute as follows: Mount Zip 1 -> Train -> Save Checkpoint -> Mount Zip 2 -> Resume -> Save Checkpoint -> Mount Zip 3 -> Save Checkpoint (Epoch 1 Complete).
*   **The Justification:** Loading all data simultaneously causes fatal I/O disk thrashing and exceeds Kaggle dataset limits. Sequential chunking perfectly mimics production-scale Deep Learning data streaming. Furthermore, aggressively saving checkpoints between every zip file guarantees that if a Kaggle session times out or crashes (a common occurrence on the free tier), zero training progress is lost.

### 2.6 The Zero-Storage Kaggle Workaround
*   **The Constraint:** Kaggle's `/kaggle/working/` local disk is hard-capped at 20GB. Attempting to extract a 34GB LINDAT zip file via `!unzip` immediately crashes the environment. Furthermore, Kaggle's automatic dataset extractor fails to unpack the LINDAT API URLs because they lack a `.zip` extension (saving them as a raw `content` blob).
*   **The Solution:** We bypass extraction entirely. We utilize Python's built-in `zipfile` library to stream the specific JSON keypoint files directly from the compressed `content` blob into RAM on the fly during the PyTorch `__getitem__` call.
*   **The Justification:** This consumes 0 bytes of disk space and eliminates hours of extraction time. We pair this with the newly discovered `YT.translations.all.json` file (bundled by the original researchers), which maps `video_id.start-end` string keys directly to English sentences, providing an O(1) lookup dictionary that perfectly matches the filenames inside the zip.

---

## 3. The Minimum Viable Product (MVP) Blueprint
To deliver a working prototype for contests and demonstrations without waiting for the full production model, we employ a tightly constrained, deterministic architecture.

### 3.1 Patient to Doctor (PSL Gestures +' Text)
*   **Methodology:** Sequence Classification.
*   **Implementation:** We define a strict vocabulary of 30-50 emergency medical phrases. We record custom videos of these phrases, extract keypoints, and train a lightweight LSTM or Spatial-Transformer classifier.
*   **Target Metrics:** Prototype demonstrations must achieve **>90% Accuracy** on the defined vocabulary, with an inference **Latency** of <2 seconds from sign completion to text display.
*   **Justification:** Full autoregressive translation requires massive data. Classification over a fixed vocabulary guarantees high accuracy, trains in minutes, and meets strict latency requirements for a live clinical demo.

### 3.2 Doctor to Patient (Voice +' Avatar Gestures)
*   **Methodology:** Trigger-Based Avatar Animation.
*   **Implementation:** We utilize OpenAI Whisper (or Google Cloud STT) for real-time Voice-to-Text transcription. The text is mapped via NLP similarity to our predefined phrases. The matched phrase ID triggers a pre-animated 3D avatar sequence.
*   **Target Metrics:** Voice-to-Text accuracy must exceed 85%, and the Avatar response **Latency** must be <1.5 seconds from the end of the clinician's speech.
*   **Justification:** Real-time generative skeletal AI suffers from severe "uncanny valley" effects and high latency. Triggering pre-rendered animations ensures immediate, professional feedback.

### 3.3 AI Safety & Escalation Framework
*   **Methodology:** Confidence Thresholding & Human Fallback.
*   **Implementation:** The patient-to-doctor classifier output is gated by a confidence score threshold. If the model's confidence in a translation falls below a clinically defined safety threshold (e.g., 85%), the system suppresses the translation.
*   **Justification:** To comply with SIMPACT AI safety expectations, low-confidence medical translations must never be presented to the clinician. The system instead outputs a fallback message ("Low Confidence: Human Interpreter Required"), ensuring patient safety and minimizing liability.

---

## 4. Current Progress & Verified Milestones

### o. Completed & Verified
- **Architectural Blueprint:** The complete pipeline (Keypoints +' Visual Encoder +' LoRA mT5) has been documented and verified to theoretically fit within Kaggle T4 parameters.
- **Dataset Storage Hack:** Successfully verified a pipeline to bypass Kaggle's 20GB local storage limit by mounting the massive LINDAT YouTube-ASL zip files directly via the "Remote Files" UI.
- **Checklist Framework:** Established a 6-phase tracker (`checklist.md`) mapping the entire pipeline from raw data to the live MVP demo.

### 🚀 Immediate Next Steps (Phase 1 Execution)
1.  Initialize the public Kaggle dataset by linking the first LINDAT URL (which downloads as the `content` blob) and the `YT.translations.all.json` file.
2.  Develop the PyTorch `Dataset` class to parse the JSON translations dictionary.
3.  Implement the `zipfile` memory-streaming logic to pair the loaded text with the compressed keypoints on the fly.
4.  Verify the pipeline by successfully extracting and structuring `dataset[0]` without triggering Kaggle OOM errors.

---
*Note: This document is strictly bounded by what is technically feasible under current hardware constraints. No assumptions are made regarding the existence of continuous PSL datasets; all PSL data required for Phase 3 and the MVP will be custom-recorded under constrained protocols.*