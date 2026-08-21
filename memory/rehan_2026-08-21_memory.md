# Memory Log: 2026-08-21 (Rehan)

## Current Context
- **Goal**: Target Pakistani Sign Language (PSL) specifically for medical (doctor-patient) conversations. Build a contest-ready Minimum Viable Product (MVP) featuring two-way translation (Patient-to-Doctor via classification, Doctor-to-Patient via Voice-to-Avatar).
- **Current Phase**: Phase 1 (Data Pipeline Execution) & Documentation.
- **Collaborator**: Coordinating with Khizer's agent via this shared repository (`exiledftw/A-PSL`).

## Notes
- Reviewed Khizer's new 6-Phase checklist system (`checklist/checklist.md` and `checklist/item_1.md`).
- Answered user questions regarding Transfer Learning (why ASL pre-training doesn't degrade PSL) and why we cannot use off-the-shelf pre-trained ASL models (heavy VLMs cause OOM, no keypoint weights released).
- Drafted and pushed a comprehensive, research-style project whitepaper (`docs/Project_Whitepaper.md`). This document consolidates all technical justifications (Keypoints vs Raw Video, Transfer Learning, LoRA + mT5) and strictly bounds the project to what is technically feasible under T4 GPU constraints, without assuming the existence of continuous PSL datasets.