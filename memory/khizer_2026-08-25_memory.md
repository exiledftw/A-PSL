# Session Log: August 25, 2026

## Work Accomplished
- **Resumed Training Successfully**: Successfully picked up from the Zip 1 deadlock (at step 1090) using the checkpoint we salvaged (`epoch_1.pt` via W&B Artifacts).
- **The "Alignment Cliff" Hit**: During early Zip 2 training, the loss unexpectedly dropped from ~8.7 down to ~4.3 for a brief moment, before returning to ~8.7. We identified this as standard "resumption spike" behavior, mixed with the model starting to align the visual embeddings to the text space.
- **Automated Checkpoint Resumption Script**: Created a script specifically for downloading checkpoints dynamically from Weights & Biases Artifacts to completely bypass local upload/download times for Kaggle.
- **W&B Secret Authentication**: Implemented `wandb.login(key="...")` directly into the background script so W&B doesn't freeze the headless container asking for a prompt.
- **Kaggle Background Execution Setup**: We finally transitioned to a "Save & Run All" background run strategy to prevent Kaggle frontend UI desyncs and to enable completely hands-free 12-hour training runs. 

## Important Context for Next Agent
- **Data Status**: We are currently running **Zip 2** in a background "Save & Run All" session. 
- **Kaggle Pitfalls Avoided**:
  - `num_workers = 0` MUST remain set to 0. Yesterday we confirmed a massive zipfile deadlock crash happens at `num_workers > 0`.
  - The W&B auto-upload script is now staged at the very bottom of the notebook. When the Kaggle background run finishes, it will automatically package the next checkpoint and upload it to W&B.
- **Current Run State**: Zip 2 training is underway. ETA is roughly 3.5 hours for the current epoch (Zip 2).

## Next Steps
- Wait for the Zip 2 background run to finish and verify that `epoch_2.pt` (or equivalent) appears in the W&B Artifacts dashboard.
- Prepare to transition to Zip 3 using the exact same W&B resumption logic.