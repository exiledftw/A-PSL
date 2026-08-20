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

While How2Sign provides clean data, it features a limited number of signers in a controlled studio environment. To make our model robust to real-world conditions, we will augment our training with the **YouTube-ASL** dataset. 

### The Streaming Storage Hack

YouTube-ASL contains thousands of hours of "in-the-wild" video, which would easily consume terabytes of storage—far exceeding the limits of Colab or Kaggle. 

To bypass this hardware constraint, we will implement an **on-the-fly streaming pipeline**:
1. Instead of downloading the massive video corpus upfront, we will only store the lightweight metadata file containing the **YouTube video IDs** and their corresponding text annotations.
2. During the training loop, our custom data loader will use the video ID to dynamically stream or temporarily download the required video chunk for the current batch.
3. Once the batch is processed (e.g., features extracted and passed through the model), the video chunk is immediately discarded from disk.

This approach ensures that our storage footprint remains virtually zero while allowing the model to train on an essentially infinite stream of diverse, real-world ASL data.