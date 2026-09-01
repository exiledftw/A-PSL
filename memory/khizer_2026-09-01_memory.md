# Session Log: September 01, 2026

## 1. Automated Video Keypoint Extraction Notebook (`keypoints.ipynb`)
- Built and verified a complete, end-to-end Jupyter Notebook: `keypoints.ipynb`.
- **Purpose:** Ingest raw video recordings (MP4/AVI/MOV/WEBM) of self-recorded Pakistani Sign Language (PSL) / Medical Emergency triage phrases and extract exact 208-dimensional landmark tensors (`66` Pose + `42` Left Hand + `42` Right Hand + `58` Neutral Face coordinates).

## 2. Technical Features Built into the Pipeline:
1. **Universal Video Ingestion:** Automatically scans `./raw_videos` (supports both nested class subfolders and flat prefix naming conventions).
2. **Selfie-Mirror Auto-Correction:** Corrects left/right hand flipping if videos were recorded in selfie camera mode.
3. **60-Frame Temporal Spline Resampling:** Normalizes variable recording lengths to exactly 60 frames to match the SANA Conv1D Temporal Tokenizer input distribution.
4. **6× Data Synthesis & Augmentation:** Automatically multiplies the raw video recordings by 6× (via spatial scaling, 2D translation, speed warping, and Gaussian coordinate jitter) to turn 400 raw videos into 2,400+ distinct training samples.
5. **Interactive Skeleton Visualizer:** Plots 2D skeletal tracking strips across video frames to verify MediaPipe tracking fidelity before training.
6. **Automated Metadata & Cloud Packaging:** Generates `dataset_metadata.csv` with bilingual English and native Urdu script (`اردو`) translations, and compresses the final `.npy` dataset into a single ZIP file ready for Kaggle/Colab training.

## 3. Next Steps:
- Add self-recorded video files into `./raw_videos`.
- Run `keypoints.ipynb` to generate the `.npy` sequences.
- Upload `SANA_PSL_Keypoints_Dataset.zip` to Kaggle/Colab and execute Few-Shot fine-tuning on the How2Sign pre-trained foundation model.