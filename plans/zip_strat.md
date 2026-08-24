# Kaggle 3-Zip Training Strategy (ASL Pre-training)

## The Core Problem
To train the Spatial-Temporal Visual Encoder (Phase 3), we need to process the full YouTube-ASL dataset (117,000 videos). The data is split across three massive `.zip` files provided by LINDAT, each weighing roughly 34GB. 

**The Blocker:** Kaggle limits attached datasets to a maximum of **100 GB** per notebook.
34 GB × 3 = 102 GB. If we try to attach all three zip files simultaneously, Kaggle's UI throws a size limit error. We can only mount one zip file at a time.

## The Pitfall: Catastrophic Forgetting
A naive solution would be to mount Zip A, train on it for 5 epochs, then mount Zip B, train for 5 epochs, etc. 

**Why we rejected this:** This causes *Catastrophic Forgetting*. If the model trains for 5 consecutive epochs on Zip A, it will severely overfit to the vocabulary and signers in that specific subset. When it switches to Zip C, it will erase its knowledge of Zip A to make room for the new data.

**The Correct Approach:** We must cycle through the zips sequentially: **Zip A → Zip B → Zip C → Zip A → Zip B...** until we complete 5 full cycles. This mimics true shuffling and ensures generalized learning.

## The Mathematical Resolution (The "15 Epoch" Trick)
Because PyTorch only sees one 39,000-video zip file at a time, it calculates the Learning Rate decay based on a 39,000-video dataset. If we set `MAX_EPOCHS = 5`, the learning rate would reach zero way too early.

**The Hack:** 
1. We set `CONFIG["MAX_EPOCHS"] = 15`.
2. PyTorch calculates `total_optimizer_steps` as `39,000 videos * 15 epochs = 585,000 total videos`.
3. This perfectly matches our true goal: `117,000 total videos * 5 epochs = 585,000 total videos`.

By lying to PyTorch, the Learning Rate decay curve is stretched out to the exact mathematical length required for our full training run.

## The Code Fix: Stopping the Loop
If `MAX_EPOCHS = 15`, PyTorch will attempt to run 15 consecutive epochs on Zip 1 before stopping. To prevent this and allow us to manually swap the zips, we added a single `break` statement to the bottom of the training loop in **Cell 9**:

```python
        # ── End-of-epoch checkpoint ──
        save_checkpoint(CHECKPOINT_PATH, epoch + 1, global_step) 
        
        # Save a permanent copy for this specific epoch so you can download it
        epoch_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"epoch_{epoch + 1}.pt")
        save_checkpoint(epoch_path, epoch + 1, global_step)
        
        break # <--- FORCES SCRIPT TO STOP AFTER 1 ZIP SO USER CAN SWAP
```

## Edge Case Resolved: Unequal Zip Sizes
**Question:** *What if Zip 1 has 39,000 videos, but Zip 2 has 42,000 videos? Does the math break when we swap zips?*

**Answer:** No, the architecture self-corrects. When the notebook restarts with Zip 2, it calculates a brand new `total_optimizer_steps` based on the 42,000 count. When we run `scheduler.load_state_dict(checkpoint)`, PyTorch remembers the exact step it is on, but it dynamically *flattens or compresses* the slope of the learning rate curve so it perfectly hits zero at the new finish line. It is mathematically safe.

---

## Step-by-Step Execution Plan

### Run 1: Zip A
1. **Attach Data:** Attach `yt-asl-1` dataset in the Kaggle sidebar. 
2. **Config (Cell 1):**
   * `"MAX_EPOCHS": 15` *(Never change this)*
   * `"DATA_PATH": "/kaggle/input/yt-asl-1/content"`
   * `"RESUME_CHECKPOINT_PATH": None`
3. **Action:** Run All.
4. **Finish:** Notebook stops after 1 pass. Download `epoch_1.pt` to local machine.

### Run 2: Zip B
1. **Swap Data:** Remove `yt-asl-1` from Kaggle sidebar. Attach `yt-asl-2`.
2. **Upload Weights:** Create a new Kaggle Dataset called `my-apsl-weights`. Upload `epoch_1.pt` to it and attach it to the notebook.
3. **Config (Cell 1):**
   * `"DATA_PATH": "/kaggle/input/yt-asl-2/content"`
   * `"RESUME_CHECKPOINT_PATH": "/kaggle/input/my-apsl-weights/epoch_1.pt"`
4. **Action:** Run All.
5. **Finish:** Notebook stops. Download `epoch_2.pt`.

### Run 3: Zip C
1. **Swap Data:** Remove `yt-asl-2`. Attach `yt-asl-3`.
2. **Upload Weights:** Update the `my-apsl-weights` Kaggle Dataset by uploading `epoch_2.pt`.
3. **Config (Cell 1):**
   * `"DATA_PATH": "/kaggle/input/yt-asl-3/content"`
   * `"RESUME_CHECKPOINT_PATH": "/kaggle/input/my-apsl-weights/epoch_2.pt"`
4. **Action:** Run All.
5. **Finish:** Notebook stops. Download `epoch_3.pt`.

### Runs 4 through 15 (The Loop)
At this point, the model has seen all 117,000 videos exactly once (True Epoch 1 completed). 
Repeat the cycle: go back to Zip 1, feed it `epoch_3.pt`, and it generates `epoch_4.pt`. Cycle A -> B -> C continuously until `epoch_15.pt` is generated.