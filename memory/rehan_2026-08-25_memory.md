# Rehan's Log - August 25, 2026

## 1. Strategic Division: Primary Training on How2Sign
Today, we made a strategic pivot to parallelize training between Khizer and me:
- **Khizer:** Handling large-scale, in-the-wild pre-training on YouTube-ASL (117k clips).
- **Rehan:** Executing full continuous ASL model training on How2Sign (31,047 clean, studio-quality frontal clips).

This dual-track approach provides immediate redundancy for our SIMPACT 2026 clinical showcase and allows us to train on a studio environment that closely mirrors clinical patient intake.

## 2. Zero-Assumption Data Discovery & Verification
We followed strict zero-assumption verification on Kaggle:
- Identified `PSewmuthu/How2Sign_Holistic` on Kaggle, containing pre-extracted MediaPipe Holistic features.
- Inspected raw `.npy` files: verified exact dimensions `(T, 543, 3)`.
- Sliced and aligned the exact 104 landmarks (33 pose + 42 hands + 29 expression face points), matching `INPUT_DIM = 208`.
- Normalization: verified torso bounding box normalization works cleanly with 0 NaNs.
- Linked `SENTENCE_NAME` directly to English text labels from `how2sign_realigned_train.csv`.

## 3. Training Architecture & Optuna Hyperparameters
Implemented `How2SignTranslator`:
- **Visual Encoder:** 2-Layer SpatialTemporal Transformer (4.3M params, 8 heads, FFN=1024, Dropout=0.108).
- **Decoder:** Google `google/mt5-small` (560M params, frozen).
- **Hyperparameters:**
  - Peak Learning Rate: `2.00e-4`
  - Weight Decay: `6.1357e-4`
  - Warmup Ratio: 5% (2,910 steps) with linear decay across 15 epochs.
  - Precision: FP32.

## 4. Weights & Biases (W&B) Integration & Cloud Checkpointing
- Connected project `sana-sign-how2sign` with run name `how2sign-visual-encoder-mt5-15ep`.
- Enabled live telemetry: step loss and learning rate streamed every 50 steps.
- Implemented automatic cloud checkpointing: on every new best validation loss, the 52.4 MB `best_how2sign_model.pt` is uploaded to W&B Artifacts (`how2sign-best-model`).

## 5. Early Training Milestones
- **Epoch 1 Completed:**
  - Initial Loss: ~22.0
  - Step Loss at step 7500: 6.70
  - **Validation Loss:** **`4.2730`** (A massive reduction compared to Optuna's 10k baseline of 7.93).
  - Epoch Duration: ~18.2 minutes (1096s).
- **Epoch 2 Running:** Peak learning rate hit at Step 6000 (`2.00e-4`) and decay began smoothly.

## 6. Artifacts & Reports Produced
- `docs/SANA_Sign_How2Sign_Training_Report.md`: Full technical whitepaper with Mermaid diagrams and mathematical proofs.
- `SANA_How2Sign_Train.ipynb`: The complete training notebook with W&B logging.
- `SANA_How2Sign_Colab_Inference.ipynb`: Standalone Google Colab tester notebook to load `best_how2sign_model.pt` and run live inference translations.
