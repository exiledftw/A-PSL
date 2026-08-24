# Rehan's Log - August 24, 2026

## 1. Optuna Sweep Execution & Troubleshooting
Today I executed the standalone Optuna sweep notebook (`SANA_Optuna_Sweep.ipynb`) on Kaggle using a 10,000 clip subset (Zip 1) to find the best hyperparameters before Khizer runs the full 15-hour training session.

Encountered and solved several Kaggle environment issues:
- **Pathing:** Fixed `IsADirectoryError` by ensuring the paths pointed to the actual files (`content` and `YT.translations.all.json`), not just their parent folders.
- **NaN Loss Explosions:** Resolved numerical instability (`val_loss=nan` and `inf`) by doing three things:
  1. Disabled `FP16` mixed precision (since we have 30GB VRAM on Kaggle's dual T4s, we don't need it and it was causing float overflows).
  2. Filtered out any clips with `NaN` keypoints during the dataset loading phase.
  3. Removed `dim_feedforward=512` from the search space to prevent collisions with `d_model=512`.

## 2. The RAM Cache Breakthrough
Initially, the Optuna trials were incredibly slow (CPU pinned at 100%, GPU at 0%). We realized the bottleneck was zip file I/O—opening a 34GB zip file thousands of times per epoch. 
**Solution:** We built a `CachedDataset` block that pre-loads all 10,000 clips directly into RAM once before the study begins. This reduced the time per trial to ~30-40 seconds, allowing 20 full trials to finish in under 10 minutes.

## 3. "Zoom In" Phase and Decay Tuning
To ensure the parameters scale properly to Khizer's full dataset run, we added `weight_decay` and `scheduler_type` (Linear vs Cosine) to the Optuna search space. After an initial broad sweep, we did a "Zoom In" run, tightly restricting the search space around the initial winning parameters to squeeze out the absolute best configuration. 

The loss successfully dropped to ~7.93, representing an 80% relative increase in confidence for the correct words compared to random initialization (12.4 loss). The data ceiling for 10k clips has been reached—the remaining loss will only drop by exposing the model to the full 117k clip corpus.

## 4. Final Handoff to Khizer
The hyperparameter search is officially complete. These are the final tuned values that Khizer must paste into `claude-a-psl.ipynb` (Cell 1) before starting the main run:
- `"LEARNING_RATE": 2.00e-04`
- `"WEIGHT_DECAY": 6.1357e-04`
- `"DROPOUT": 0.108`
- `"NUM_ENCODER_LAYERS": 2`
- `"DIM_FEEDFORWARD": 1024`
- `"BATCH_SIZE": 4`
- `"GRADIENT_ACCUMULATION_STEPS": 2`
- `"SCHEDULER_TYPE": "linear"`
- `"WARMUP_STEPS": 3343` *(Scaled for 15 epochs)*

## 5. Conceptual Alignment
Re-aligned on the model's core purpose for the SIMPACT showcase: We are not just mapping gestures to glosses. We are using the Visual Encoder (the part we just tuned) as a "bridge" to map physical ASL keypoint physics directly into the semantic embedding space of a frozen mT5 language model, allowing the system to output coherent English sentences directly from video.