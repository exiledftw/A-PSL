# Memory Log: 2026-08-21 (Rehan)

## Current Context
- **Goal**: Target Pakistani Sign Language (PSL) specifically for medical (doctor-patient) conversations. Build a contest-ready Minimum Viable Product (MVP) featuring two-way translation (Patient-to-Doctor via classification, Doctor-to-Patient via Voice-to-Avatar).
- **Current Phase**: Phase 1 (Data Pipeline Execution) & Documentation.
- **Collaborator**: Coordinating with Khizer's agent via this shared repository (`exiledftw/A-PSL`).

## Notes
- Reviewed Khizer's new 6-Phase checklist system (`checklist/checklist.md` and `checklist/item_1.md`).
- Answered user questions regarding Transfer Learning (why ASL pre-training doesn't degrade PSL) and why we cannot use off-the-shelf pre-trained ASL models (heavy VLMs cause OOM, no keypoint weights released).
- Drafted and pushed a comprehensive, research-style project whitepaper (`docs/Project_Whitepaper.md`). This document consolidates all technical justifications (Keypoints vs Raw Video, Transfer Learning, LoRA + mT5) and strictly bounds the project to what is technically feasible under T4 GPU constraints, without assuming the existence of continuous PSL datasets.
- Progressed on Checklist Tasks 1.1 and 1.2: Initiated download of YT-ASL zip files directly to Kaggle using the Remote URL feature (bypassing local download/upload times).
- **Technical Pivot (Data Loading):** Since Kaggle parses the LINDAT DSpace API URLs without a `.zip` extension and downloads them as a raw compressed `content` blob, we have decided *not* to force Kaggle to extract them. Instead, we will use Python's built-in `zipfile` library to stream the JSON keypoint files directly from the compressed archive in memory during training. This prevents OOM errors on Kaggle's limited 20GB local disk and dramatically speeds up Kaggle dataset loading.
- **SIMPACT 2026 Alignment:** Received formal project requirements for the SIMPACT 2026 showcase (CIME Karachi, Sept 17, 2026). The project is officially branded "SANA Sign" (part of SANA AI HIMS). I entered planning mode and updated the Whitepaper, Strategy 3, and Checklist files to integrate the required deliverables: a strict Safety Framework (Confidence threshold + human fallback message), specific metrics tracking (Accuracy >90%, Latency <2s), and documentation deliverables (Patient consent form, clinical validation design, avatar user testing). Also restored the corrupted header in `item_1.md`.