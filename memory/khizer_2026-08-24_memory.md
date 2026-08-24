# Session Memory: Khizer
**Date:** 2026-08-24

## Context
Working on the A-PSL Phase 1-3 pipeline (YouTube-ASL foundation) for the SIMPACT 2026 showcase MVP.

## Tasks Completed
1. **Evaluated Generated Notebooks:** Compared the Bolt vs. Claude Kaggle notebooks. Selected the Claude notebook for its superior robustness (OOM auto-recovery, dynamic batch padding, schema-aware parsing).
2. **Kaggle Directory vs Zip Compatibility:** Updated the Dataset logic to automatically handle loading keypoints from either an unzipped directory (for 100-video tests) or a massive `.zip` blob (for the 34GB real run) using a `DATA_IS_ZIP` toggle.
3. **Pipeline Smoke Test:** Executed a successful 1.5-minute test training run on Kaggle using 100 videos.
4. **Weights & Biases Integration:** Added W&B tracking for live VRAM peaks, learning rates, loss curves, and sample prediction tables.
5. **Fixed Missing Landmark Crash:** Updated the `select_keypoints` function to gracefully zero-pad when MediaPipe fails to detect a face or body pose on specific frames, preventing data-loading crashes on real-world noisy clips.
6. **Inference Validation:** Confirmed the frozen mT5 model successfully outputs multilingual text (which looks like gibberish prior to full training, as expected).
7. **Per-Epoch Checkpointing:** Modified the training loop to save permanent, uniquely-named checkpoint files (e.g., `epoch_1.pt`) instead of just overwriting a single `latest.pt` file, allowing easy download/export and safe resumption after disconnects.
8. **Updated Checklist:** Marked Phase 1 and 2 items as completed, and Phase 3 as mostly completed/in-progress in `checklist/item_1.md`.

## Notes for Next Session
- **Immediate Next Step:** Switch `DATA_PATH` to the full 34GB Kaggle zip (`/kaggle/input/datasets/kkmalik/yt-asl/content`), set `MAX_CLIPS: None`, and begin the full ASL Pre-training run (Phase 3).
- **Deployment Note:** The `.pt` weights generated here will eventually be exported to **ONNX** format for the high-speed, low-latency live demo requirement of SIMPACT 2026 (llama.cpp/GGUF is incompatible with our custom spatial-temporal encoder).
- **Checkpoint Resumption:** If Kaggle disconnects, upload the latest `epoch_X.pt` file as a Kaggle dataset, point `RESUME_CHECKPOINT_PATH` to it, and re-run. The script will dynamically pick up exactly where it left off.