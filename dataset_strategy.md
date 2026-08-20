# Dataset Strategy: Sign Language to English Translation

This document outlines the dual-dataset strategy for training our continuous American Sign Language (ASL) to English translation model, optimized for compute environments with strict storage limits (e.g., Google Colab, Kaggle).

## 1. Primary Dataset: How2Sign

**Role:** The core foundational dataset for model training.

How2Sign is our primary dataset because it offers a highly structured, "gold standard" continuous ASL corpus. It provides:
- **Pre-extracted MediaPipe keypoints**, which saves us from having to run heavy pose-estimation models on raw video during the initial training phases.
- **Gloss annotations**, which are critical for our two-stage training approach (Video → Gloss → English).

Since the pre-extracted keypoints for How2Sign are relatively lightweight (~20-30 GB), we can comfortably download and store them directly in our training environment.

## 2. Secondary / Augmentation Dataset: YouTube-ASL

**Role:** The supplementary dataset to improve real-world generalization and signer diversity.

**Source:** The pre-extracted keypoint version of YouTube-ASL is hosted on LINDAT/CLARIAH-CZ digital library by Zelezny, Hruz, Straka, and Gueuwou (2024). Permanent handle: http://hdl.handle.net/11234/1-5898. The dataset page is at https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5898.

**Format:** 390,547 JSON files containing frame-by-frame 2D keypoints extracted using MediaPipe, generating 208 2D keypoints per frame representing body, face, hands, and pose landmarks. Files are distributed across 10 separate zip files for easier downloading.

**Access strategy:** Do not download videos. Do not run MediaPipe extraction. Instead, obtain the 10 direct .zip download URLs from the LINDAT file listing page. Create a new public Kaggle dataset using Kaggle's "New Dataset → Remote Files" UI feature — paste each zip URL so Kaggle's servers pull the files directly from LINDAT into Kaggle dataset storage. No local bandwidth or notebook scratch space is consumed during this process.

**Storage:** The dataset lives in Kaggle's dataset storage (200GB limit), not the notebook's working directory (20GB limit). Once uploaded publicly, mount it in any training notebook via "Add Data" in the notebook sidebar. It will appear at `/kaggle/input/<dataset-name>/` and is readable directly without copying into working space. Both team members (Khizer and Rehan) can mount and use the same public dataset.

**Caption pairing:** The keypoint JSON files do not contain English translations. Captions must be obtained separately from the YouTube-ASL metadata TSV available on Google's GitHub (google-research/google-research/youtube_asl). Each JSON clip is matched to its English caption by joining on the clip identifier (`video_id` + `start_timestamp` + `end_timestamp`) as the key.

**Pre-training check:** Before launching training, verify the JSON structure of a sample file to confirm exact field names (`keypoints`, `video_id`, `start`, `end`) and confirm the clip identifier format matches the YouTube-ASL captions TSV. Adjust the Dataset class join key accordingly.

**Why this approach:** Avoids all video downloading, proxy usage, yt-dlp rate limiting, and local storage pressure. Kaggle free tier's 20GB scratch space is preserved entirely for model checkpoints and outputs. Dataset persists permanently across sessions — no re-downloading on session restart.