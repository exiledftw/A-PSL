# Session Log: August 27, 2026

## Work Accomplished
- **Epoch 2 Kickoff**: We officially began True Epoch 2 (looping back to Zip 1) to continue pre-training the model.
- **W&B Checkpoint Path Bug Fix**: The background runner crashed at the start of Epoch 2 because W&B dynamically changed the downloaded folder version (e.g., from `v1` to `v2`). We patched the Kaggle notebook by implementing a `glob` snippet that searches the `/kaggle/working/artifacts/` directory and automatically links the newest `.pt` file to the config. This permanently prevents path mismatches.
- **Accuracy Tracking Added**: We updated the custom `run_train_step_with_oom_backoff` function to passively calculate token-level accuracy during the forward pass (using `torch.no_grad()`). It is now safely averaging across gradient accumulation steps and successfully logging to both the Kaggle console and the W&B dashboard as `train/accuracy`.

## Strategic Pivots (CEO Directive)
- **Phase 5 Modification (Preventing Data Leakage)**: The CEO correctly identified that recording a custom PSL dataset using the exact same people who will present the live SIMPACT 2026 demo constitutes severe data leakage. 
- **Action Taken**: We scrapped the plan to record our own 750-clip dataset. We are now officially pivoting to partner with an organization to acquire a diverse, medium-sized PSL dataset. This ensures the final model will prove true generalization. 
- **Repo Updated**: I updated the Mermaid graph and the Phase 5 sub-tasks in `checklist/item_1.md` to reflect this new organizational partnership strategy.

## Technical Insights & Debugging
- **Loss Logging Glitch**: Identified a classic PyTorch checkpoint resumption glitch where the very first W&B loss log appeared exactly halved (e.g., 4.25 instead of 8.5). This occurs because `running_loss` accumulates for only a partial interval (e.g., 20 batches) upon resumption but still divides by the hardcoded `LOG_EVERY_N_STEPS` denominator (40 batches). It is purely visual.
- **Vocabulary Size vs Dataset Size**: Realigned expectations regarding the loss. A loss of 8.5 is incredibly healthy for early Epoch 2 given the mathematical scale: the model is attempting to map noisy "YouTube-in-the-wild" keypoints to a massive `google/mt5-small` dictionary of 250,112 tokens, using only a subset of 117k examples. The "Aha!" moment typically requires multi-epoch repetition.
- **Hardware Viability (Local vs Kaggle)**: Evaluated local training on an RTX 3060 Laptop GPU (6GB VRAM) vs Kaggle's T4 (15GB VRAM). Concluded that 6GB is a strict bottleneck for Phase 3 (Training) due to Optimizer states + 560M params taking >5GB, leading to immediate OOM crashes even at micro-batch size 1. However, the laptop will be perfect for Phase 6 (Inference) where VRAM footprint is minimal.

## Next Steps
- Monitor the `train/accuracy` metric on W&B during the current Zip 1 run.
- Continue cycling through the zip files until the Phase 3 pre-training completes.
- Begin outreach to organizational partners for the Phase 5 PSL dataset.