# Strategy 1: Visual-Encoder + T5-Decoder with LoRA (YT-ASL → How2Sign)

## 1. Overview
This document outlines an advanced, state-of-the-art training paradigm for our ASL-to-English translation model. It borrows concepts from modern Vision-Language Models (VLMs) by leveraging massive, noisy data for pre-training, followed by Low-Rank Adaptation (LoRA) fine-tuning on clean, curated data.

## 2. The Modality Gap & The "T5 Catch"
We want to utilize **T5 (Text-to-Text Transfer Transformer)** as our foundational backbone because it already possesses a flawless understanding of English grammar. 

**The Challenge:** T5 expects discrete text tokens as input. Our dataset consists of continuous spatial-temporal floating-point numbers (MediaPipe 2D keypoints: `T frames × 75 landmarks × 3 coordinates`). We cannot feed `.npy` floats directly into T5's embedding matrix.

**The Solution:** We must build a **Cross-Modal Architecture**:
1. **Visual Encoder:** A lightweight Spatial-Temporal Transformer designed specifically to process MediaPipe landmarks and generate continuous "Visual Embeddings."
2. **Projection Layer:** A linear mapping layer that translates our Visual Embeddings into the exact latent dimensionality that the T5 Decoder expects.
3. **T5 Decoder:** The pre-trained LLM backbone that receives the projected visual embeddings and autoregressively generates the English translation.

## 3. The Two-Phase Training Strategy

### Phase 1: Pre-training (Massive Scale, High Noise)
- **Dataset:** YouTube-ASL (600k+ in-the-wild clips).
- **Goal:** Teach the Visual Encoder the broad strokes of ASL gestures, alignments, and representations across diverse signers and noisy environments.
- **Mechanics:** 
  - We freeze the heavy T5 Decoder completely to save VRAM.
  - We compute loss and update weights **only** for the Custom Visual Encoder and the Projection Layer.

### Phase 2: Fine-Tuning (Curated Scale, Low Noise)
- **Dataset:** How2Sign (35k studio-quality clips with gloss annotations).
- **Goal:** Perfect the translation accuracy, grammar alignment, and task-specific performance without catastrophic forgetting of the real-world diversity learned in Phase 1.
- **Mechanics:**
  - We inject **LoRA (Low-Rank Adaptation) adapters** into the self-attention blocks of the T5 Decoder.
  - We unfreeze the Visual Encoder and train it alongside the newly added LoRA adapters.
  - We do *not* train the base weights of the T5 model, ensuring compute efficiency.

## 4. Hardware & T4 Compute Advantages
This strategy is highly optimized for the 15GB VRAM limit of a Google Colab / Kaggle T4 GPU:
- **T5 Pre-training Avoided:** Training a full LLM on a T4 leads to Out-Of-Memory (OOM) crashes. Freezing it in Phase 1 solves this.
- **LoRA Efficiency:** LoRA reduces the number of trainable parameters in the Decoder by over 99%, keeping optimizer states and gradient memory extremely small during Phase 2.
- **Language Free-Riding:** Because T5 already knows English perfectly, the model spends its compute budget learning the *alignment* between ASL and English, rather than wasting epochs learning how to construct a valid English sentence.