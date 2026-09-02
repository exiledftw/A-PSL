# SANA A-PSL — Live Webcam Inference: Handoff Document

## TL;DR

The trained model (`SANA_PSL_Translator`) works. Live webcam predictions were failing/looked untrustworthy at first — not because the model was broken, but because the **live inference pipeline wasn't reproducing the exact preprocessing the model was trained on**. Once the webcam script matched the training pipeline byte-for-byte (same landmarks, same order, same normalization, same resampling/padding), predictions became reliable and testably not a fixed fallback output.

This doc explains what was actually wrong, what we checked before assuming anything, what the fix was, and how to validate the model yourself going forward.

---

## 1. The Core Problem

We had two separate notebooks that needed to agree perfectly on data format, but nothing in the repo connected them explicitly:

| Notebook | Role |
|---|---|
| `notebook399144a5d3.ipynb` | Defines the model architecture (`SANA_PSL_Translator`) and trains/fine-tunes it on pre-made `.npy` files. Expects input shape `(100, 208)` per sample. |
| `keypoints.ipynb` | The actual feature extraction pipeline: takes video, runs it through MediaPipe, and produces the `.npy` files that feed the training notebook. |

**A live webcam script has to replicate `keypoints.ipynb`'s exact extraction logic**, because that's the only ground truth for what the model actually learned to interpret. Guessing at the landmark layout (e.g., assuming a generic MediaPipe Holistic setup) would silently feed the model a differently-shaped, differently-ordered, or differently-normalized vector than it was trained on — producing garbage or misleading predictions with no obvious error message.

We treated "don't assume anything" as a hard requirement here: before writing any inference code, we read both notebooks in full to extract the *exact* preprocessing contract.

---

## 2. The Exact Contract We Reverse-Engineered (not guessed)

### 2.1 The 208-dim feature vector, per frame

```
[0:66]    33 Pose landmarks   × (x, y)   -> 66 values
[66:108]  21 Left Hand landmarks × (x, y) -> 42 values
[108:150] 21 Right Hand landmarks × (x, y) -> 42 values
[150:208] 58 zeros (face slots — never populated; SANA standard placeholder)
```

### 2.2 Extraction details that had to match exactly
- **MediaPipe Tasks API**, not the legacy `mp.solutions` API — specifically `HandLandmarker` (`num_hands=2`) and `PoseLandmarker`, using the same model files (`hand_landmarker.task`, `pose_landmarker_lite.task`).
- **Confidence thresholds**: `min_detection_confidence=0.5`, `min_tracking_confidence/presence_confidence=0.5`.
- **Mirror correction**: `keypoints.ipynb` swaps left/right hand coordinates after detection (`MIRROR_CORRECTION=True`), because the source training videos were recorded in a way that made MediaPipe's raw handedness label backwards. The live script defaults to the same swap, with a runtime toggle (`m` key) since a live webcam's mirroring behavior can differ from a phone's front camera and needs visual verification, not assumption.
- **Temporal smoothing**: exponential smoothing per coordinate, `alpha=0.75`, applied only to non-zero (i.e. detected) values. This had to be replicated frame-by-frame identically, including...
- **`reset_tracker()` per clip**: the extraction pipeline resets its smoothing state at the start of every new video. The live script replicates this by resetting the tracker every time a new recording starts (SPACE key), so smoothing never bleeds between two unrelated signs.

### 2.3 Sequence length handling
- `keypoints.ipynb` resamples every raw video clip to a fixed **60 frames** (`TARGET_FRAMES=60`) via linear interpolation per-dimension (`resample_sequence`).
- The training notebook's `MedicalDataset` then **zero-pads every sample up to `MAX_SEQ_LEN=100`** (since 60 < 100) before feeding it to the model.
- **This two-step process (resample to 60 → zero-pad to 100) is what the model actually saw during training** — not a single resample-to-100 step. Getting this wrong (e.g. resampling straight to 100) would shift every gesture's temporal position relative to what the model learned, likely degrading predictions in a way that's hard to diagnose from the outside.

The live script performs the identical two-step process on every captured gesture before inference.

---

## 3. Why We Couldn't Just "Assume It'd Probably Work"

Two independent, compounding risk factors made blind testing unreliable, and we called both out before testing live:

1. **The reported "~100% accuracy" was measured on training data, not a held-out test set.** The evaluation cells in `notebook399144a5d3.ipynb` sample directly from `med_dataset` (the training set), and the config even labels the run a `"40-video overfitting test"`. That number tells us the model can memorize, not that it generalizes — so the webcam test was the *actual* first generalization test, not a confirmation of one.

2. **Small seq2seq models trained on tiny datasets are prone to output collapse** — settling on one "safe" sentence regardless of input, especially with greedy decoding (`model.mt5.generate()` uses greedy decoding by default here). This needed to be actively ruled out, not assumed away.

---

## 4. The Diagnostic We Built to Rule Out Collapse (before trusting any prediction)

A built-in **collapse diagnostic** (`d` key in the script) feeds three deliberately different inputs through the same trained model and prints all three outputs:
- The real captured landmark sequence
- An all-zeros sequence (no signal)
- A random-noise sequence

**Interpretation rule:** if all three come back identical, the model isn't conditioning on the input at all — it's a fixed fallback, and no amount of correct signing will fix it without retraining. If they differ, the model is at least reactive to input — necessary but not sufficient evidence of correctness.

### What we actually observed in testing (see full session log in `test_logs/` if committed)
- All-zeros input consistently returned **"Test are cheap here"** across multiple independent runs (including after a full script restart) — this is the model's degenerate-input fallback and should be recognized as such, not mistaken for a real prediction if it ever shows up unexpectedly.
- Random-noise input returned incoherent/sentinel output (`<extra_id_0>`, `allahu n`) — consistent with a model actually processing structured input rather than free-associating.
- **Real captured signs across 4 different phrase classes, repeated multiple times each, at varying raw frame counts (33–91 frames), correctly and distinctly mapped to their intended sentences** — including two captures with the *same* raw frame count (35 frames) producing two different, correct sentences. That last point matters: it rules out the model taking a shortcut based on something crude like sequence length, and supports it actually discriminating based on gesture shape.

**Bottom line from testing:** the model is not collapsed, and it correctly distinguished all 4 trained phrase classes across repeated live trials. This is real (if still limited-scope) evidence the pipeline alignment fix worked.

---

## 5. Known Limitations (be upfront with the team about these)

- **Training data was small (~40 videos)** and augmentation only covered scale/shift/speed/jitter — not camera distance, angle, background, or a different signer. Expect possible degradation if testing conditions (lighting, framing, distance from camera) differ significantly from how the original training videos were recorded. This hasn't been stress-tested yet.
- **"Test are cheap here" is a fallback attractor for weak/ambiguous input.** If a live capture is poor quality (hands leaving frame early, bad lighting, gesture cut short), the model may return this phrase even though it's not what was signed. Treat unexpected occurrences of this specific output with suspicion and recapture rather than trusting it at face value.
- **Greedy decoding** is used for generation (no beam search / sampling). This was left as-is since testing showed it isn't causing collapse in practice, but if the team scales to more phrase classes, revisit this — collapse risk increases with more classes and thinner data per class.
- Only 4 phrase classes have been validated end-to-end. Do not assume this generalizes to phrases/signs outside the current `medical_dictionary` without similar testing.

## 6. How to Run It (for teammates)

**Requirements:** Python 3.9+, a machine with a physical webcam (not a cloud notebook like Kaggle/Colab).

```bash
pip install mediapipe opencv-python torch transformers numpy
```

1. Place `webcam_inference.py` and the fine-tuned checkpoint (e.g. `sana_psl_medical_finetuned.pt`) in the same folder, or note the checkpoint's path.
2. Run:
   ```bash
   python webcam_inference.py --model sana_psl_medical_finetuned.pt
   ```
   First run auto-downloads two small MediaPipe model files (`hand_landmarker.task`, `pose_landmarker.task`) — needs internet once.
3. Grant camera permission if your OS prompts for it.

**Controls:**
| Key | Action |
|---|---|
| `SPACE` | Start recording a sign; press again to stop and run inference |
| `m` | Toggle mirror-hand-correction (check the on-screen "Detected: Left/Right" labels — if your real right hand shows as "Left", toggle this) |
| `d` | Run the collapse diagnostic on the last captured sign |
| `q` | Quit |

**Recommended first-run checklist for anyone new to this:**
1. Record any sign, confirm a prediction appears.
2. Press `d` immediately after — confirm the three diagnostic outputs are not all identical.
3. Test each phrase in `medical_dictionary` 3–4 times each before trusting results.
4. If "Test are cheap here" shows up unexpectedly, treat it as a likely bad-capture artifact and retry rather than a real answer.

## 7. Contact / Ownership Notes

- Model architecture source of truth: `notebook399144a5d3.ipynb`
- Feature extraction source of truth: `keypoints.ipynb`
- Both are the authoritative references — if either changes (e.g. different `INPUT_DIM`, different `TARGET_FRAMES`, different landmark set), `webcam_inference.py`'s `CONFIG` dict and extraction logic **must be updated to match exactly**, or live inference will silently misalign again. There is no automated check for this today — it's a manual contract between the three files.
