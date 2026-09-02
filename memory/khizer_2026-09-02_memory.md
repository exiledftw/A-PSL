# Khizer's Memory: 2026-09-02 (Live Webcam Inference Breakthrough)

## 1. Context & The Core Issue
We spent the session trying to get the fine-tuned SANA model (sana_psl_medical_finetuned.pt) to correctly predict phrases on a live webcam feed. Initially, the model kept returning default fallback phrases ("There has been an accident" or <extra_id_0>) across multiple iterations of the live script (live_webcam_translator.py).

## 2. The Breakthrough (Sonnet's Fix)
Claude (Sonnet) successfully audited the pipeline and identified that the live inference pipeline was fundamentally misaligned with the Kaggle training pipeline (keypoints.ipynb and 
otebook399144a5d3.ipynb). 

**The Missing Temporal Alignment:**
My previous webcam script was zero-padding the live webcam frames directly up to 100. However, the actual Kaggle dataset generation did a **two-step process**:
1. It mathematically resampled/squeezed every video to *exactly* 60 frames (TARGET_FRAMES=60).
2. It *then* zero-padded the remaining 40 frames to reach 100.

Because the live script skipped the 60-frame resample step, the temporal frequency of the live gestures completely mismatched what the 1D Convolutions in the AI learned during training.

## 3. The New Inference Pipeline (webcam_inference.py)
Sonnet wrote a new, mathematically rigorous script (webcam_inference.py) that fixes all pipeline mismatches:
- It correctly implements the esample_to_60 -> pad_to_100 logic.
- It uses the modern **MediaPipe Tasks API** (HandLandmarker + PoseLandmarker) side-by-side to perfectly recreate the [66 pose, 42 L-hand, 42 R-hand, 58 face] 208-dimensional feature vector.
- It includes a **Collapse Diagnostic (press 'd')** which proves that the model predicts correctly on real input, while gracefully falling back to "Test are cheap here" on all-zeros (no signal).

## 4. Next Steps for Khizer
- **Use webcam_inference.py** as the definitive source of truth for live inference testing.
- Review HANDOFF.md for a complete technical breakdown of the extraction contract.
- Note that any future changes to INPUT_DIM, TARGET_FRAMES, or landmark architecture in the Kaggle notebooks must be manually mirrored in webcam_inference.py to prevent silent misalignment.
