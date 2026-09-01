# Session Log: September 01, 2026

## 1. Automated Video Keypoint Extraction Notebook (`keypoints.ipynb`)
- Built and verified a complete, end-to-end Jupyter Notebook: `keypoints.ipynb` optimized for **Google Colab**.
- **Purpose:** Ingest raw video recordings (MP4/AVI/MOV/WEBM) of self-recorded Pakistani Sign Language (PSL) / Medical Emergency triage phrases directly from Google Drive and extract exact 208-dimensional landmark tensors (`66` Pose + `42` Left Hand + `42` Right Hand + `58` Neutral Face coordinates).

## 2. Google Colab & GDrive Integration:
1. **Google Drive Auto-Mount:** Uses `google.colab.drive` to mount `/content/drive`.
2. **Directory Scanner (`Test Data/{label}/[videos]`):** Auto-detects the `Test Data` folder in `MyDrive` and parses all label subfolders and raw video files.
3. **Selfie-Mirror Auto-Correction:** Corrects left/right hand flipping if videos were recorded with front-facing cameras.
4. **60-Frame Temporal Spline Resampling:** Linearly normalizes all recordings to 60 frames for the SANA Conv1D Temporal Tokenizer.
5. **6× Data Augmentation:** Multiplies raw samples by 6× (spatial zoom, position shift, temporal speed warping, and coordinate jitter).
6. **Automatic GDrive Cloud Backup & ZIP Export:** Saves processed `.npy` files locally in `/content/processed_psl_dataset` for fast processing and creates a permanent ZIP backup in `/content/drive/MyDrive/SANA_PSL_Keypoints_Dataset.zip`.
7. **Skeleton Visualizer:** Displays 2D skeletal tracking strips across multiple frames.

## 3. Next Steps:
- Open `keypoints.ipynb` in Google Colab.
- Run all cells to process the `Test Data/{label}/[videos]` directory.
- Download or use `SANA_PSL_Keypoints_Dataset.zip` for Few-Shot fine-tuning on the How2Sign pre-trained model.
## 4. Fixes
- Fixed a MediaPipe import error on Colab ('AttributeError: module mediapipe has no attribute solutions') by pinning the installation to mediapipe==0.10.14 in the uild_keypoints_nb.py script. Regenerated and pushed the updated keypoints.ipynb.
