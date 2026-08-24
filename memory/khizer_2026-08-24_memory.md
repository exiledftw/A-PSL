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
9. **Finalized Kaggle Multi-Zip Strategy:** Designed the 15-epoch manual-swap training strategy to bypass Kaggle's 100GB limit while preserving PyTorch learning rate math.

## Notes for Next Session / Other Agents
- **Deployment Note:** The `.pt` weights generated here will eventually be exported to **ONNX** format for the high-speed, low-latency live demo requirement of SIMPACT 2026 (llama.cpp/GGUF is incompatible with our custom spatial-temporal encoder).
- **CRITICAL: The 3-Zip / 15-Epoch Training Strategy:**
  Because Kaggle limits inputs to 100GB, we cannot mount all three 34GB LINDAT zips at once. The user must train using a manual swap cycle (Zip 1 -> Zip 2 -> Zip 3 -> Zip 1...) to achieve 5 true epochs over the whole dataset.
  - **The Math Hack:** We set `CONFIG["MAX_EPOCHS"] = 15`. This tricks PyTorch into stretching the learning rate decay curve over 15 runs of a single zip (which mathematically equals 5 runs of 3 zips).
  - **The Code Hack:** We added a `break` statement at the very end of the `for epoch in epoch_bar:` loop in Cell 9 (right after `save_checkpoint(epoch_path...)`). This forces the script to gracefully stop after exactly 1 zip pass, rather than spinning 15 times on the same zip.
  - **The Workflow:** 
    1. Attach Zip 1. Set `MAX_EPOCHS=15`, `RESUME_CHECKPOINT_PATH=None`. Run. 
    2. Script finishes, saves `epoch_1.pt`, hits `break`, and stops.
    3. User downloads `epoch_1.pt`, removes Zip 1 from Kaggle, attaches Zip 2.
    4. User uploads `epoch_1.pt` as a Kaggle dataset, points `RESUME_CHECKPOINT_PATH` to it. Run.
    5. Script resumes at step 2, adjusts the learning rate curve slightly if the new zip has a different length, finishes Zip 2, saves `epoch_2.pt`, hits `break`, and stops.
  - *If another agent picks this up:* Do not touch `MAX_EPOCHS` (leave it at 15), do not change the `break` statement, and do not manually change the `LEARNING_RATE`. The PyTorch scheduler handles the math automatically via `scheduler.load_state_dict`.