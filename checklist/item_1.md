```mermaid
graph TD
    subgraph DATA["dY" Phase 1: Data Pipeline"]
        A1["LINDAT YouTube-ASL\n390k JSON keypoint files\n(10 zip files)"] --> A2["Kaggle Remote Files UI\nMount at /kaggle/input/"]
        A2 --> A3["Downsample to 100k clips\n(Demographically balanced)"]
        A3 --> A4["LINDAT Translations JSON\n(Direct pairing via\nvideo_id.start-end keys)"]
    end

    subgraph MODEL["dY  Phase 2: Architecture Assembly"]
        B1["Clone T5_for_SLT repo\n(Zelezny et al.)"] --> B2["Visual Encoder\n(Spatial-Temporal Transformer)"]
        B1 --> B3["Projection Layer\n(Linear bridge)"]
        B1 --> B4["mT5 Decoder\n(Frozen, HuggingFace)\nSupports English + Urdu"]
        B2 --> B3 --> B4
    end

    subgraph TRAIN["dYs? Phase 3: ASL Pre-training"]
        C1["Freeze mT5\nTrain Visual Encoder\non 100k YT-ASL"] --> C2{"Loss decreasing?\nVRAM under 15GB?"}
        C2 -->|Yes| C3["Save checkpoint"]
        C2 -->|OOM| C4["Reduce batch size\nor drop face landmarks"]
        C4 --> C1
    end

    subgraph FINETUNE["dYZ_ Phase 4: How2Sign Fine-tune"]
        C3 --> D1["Enable LoRA adapters\non mT5 decoder"]
        D1 --> D2["Fine-tune on\nHow2Sign 35k clips"]
        D2 --> D3["Evaluate BLEU-4\non test set"]
    end

    subgraph PSL["dYdY Phase 5: PSL Pivot (Strategy 2)"]
        D3 --> E1["Scrape psl.org.pk\nPSL dictionary videos"]
        E1 --> E2["Extract MediaPipe keypoints\nfrom PSL videos"]
        E2 --> E3["Domain-adapt Visual Encoder\nfrom ASL to PSL"]
        E3 --> E4["Record 30-50 custom\nmedical PSL phrases"]
        E4 --> E5["LoRA fine-tune on\nmedical PSL dataset"]
    end

    subgraph MVP["dY?+ Phase 6: Contest MVP (Strategy 3)"]
        E5 --> F1["Patient +' Doctor:\nPSL Classification\n(LSTM/Transformer)"]
        E5 --> F2["Doctor +' Patient:\nWhisper +' Text +' Avatar"]
        F1 --> F3["Live two-way demo\non webcam"]
        F2 --> F3
    end

    DATA --> MODEL --> TRAIN

    style DATA fill:#1a2e1a,stroke:#4aff9a,color:#fff
    style MODEL fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style TRAIN fill:#2e1a1a,stroke:#ff4a4a,color:#fff
    style FINETUNE fill:#2e2e1a,stroke:#ffff4a,color:#fff
    style PSL fill:#1a2e2e,stroke:#4affff,color:#fff
    style MVP fill:#2e1a2e,stroke:#ff4aff,color:#fff
```

---

## Detailed Task Checklist

### Phase 1: Data Pipeline (YouTube-ASL)

- [x] **1.1** Obtain the 10 direct `.zip` download URLs from the LINDAT file listing page
- [/] **1.2** Create a public Kaggle dataset using "New Dataset +' Remote Files" ?" paste each zip URL so Kaggle pulls files from LINDAT directly
- [ ] **1.3** Verify the Kaggle dataset mounts correctly at `/kaggle/input/<dataset-name>/`
- [ ] **1.4** Download the `YT.translations.all.json` file directly from LINDAT
- [ ] **1.5** Write a PyTorch Dataset class to parse the LINDAT translations JSON and dynamically load the corresponding MediaPipe keypoint files using the `clip_order` keys
- [ ] **1.6** Verify the join ?" spot-check 50 random clips to confirm caption alignment
- [ ] **1.7** Select exactly 3 LINDAT zip files (~117k clips) to serve as our complete prototype dataset (achieving the ~100k target)
- [ ] **1.8** Inspect sample JSON structure ?" confirm field names and 208 landmarks per frame

### Phase 2: Architecture Assembly

- [ ] **2.1** Fork/Clone the `zeleznyt/T5_for_SLT` repository
- [ ] **2.2** Read and understand the existing data loader, model architecture, and training loop
- [ ] **2.3** Modify the decoder to use `google/mt5-small` (from HuggingFace) instead of standard T5 ?" this gives us English + Urdu output capability for the PSL pivot later
- [ ] **2.4** Verify the Projection Layer dimensions match between Visual Encoder output and mT5 embedding input
- [ ] **2.5** Inject LoRA adapter config (using `peft` library) into mT5 decoder attention blocks ?" keep disabled for Phase 3
- [ ] **2.6** Run a single forward pass with a dummy batch to confirm tensor shapes and VRAM usage

### Phase 3: ASL Pre-training (The Proof)

- [ ] **3.1** Upload training script to a Kaggle notebook with T4 GPU enabled
- [ ] **3.2** Mount the 100k dataset and the captions TSV
- [ ] **3.3** Run pre-training (Epoch 1): freeze mT5, train Visual Encoder sequentially on Zip 1 -> Checkpoint -> Zip 2 -> Checkpoint -> Zip 3
- [ ] **3.4** Monitor: Does loss decrease over 5 epochs? +' If yes, model is learning
- [ ] **3.5** Monitor: Does VRAM stay under 15GB? +' If no, reduce batch size or drop face landmarks
- [ ] **3.6** Save checkpoint to Google Drive / Kaggle output

### Phase 4: How2Sign Fine-tune

- [ ] **4.1** Download How2Sign pre-extracted keypoints (Kaggle/HuggingFace)
- [ ] **4.2** Enable LoRA adapters on mT5 decoder, unfreeze Visual Encoder
- [ ] **4.3** Fine-tune on How2Sign 35k clips (with gloss annotations)
- [ ] **4.4** Evaluate: compute BLEU-4 on How2Sign test set
- [ ] **4.5** Qualitative check: manually read 20 model outputs ?" do they make sense?

### Phase 5: PSL Domain Adaptation (Strategy 2)

- [ ] **5.1** Write a scraper for `psl.org.pk` to download PSL dictionary videos (prioritize medical category)
- [ ] **5.2** Run MediaPipe extraction on scraped PSL videos +' save as JSON keypoints
- [ ] **5.3** Domain-adapt the Visual Encoder on the PSL dictionary dataset (continue training)
- [ ] **5.4** Define 30-50 critical medical PSL phrases for the MVP
- [ ] **5.5** Record 2-3 people signing each phrase (10-15 variations each) +' ~750 clips
- [ ] **5.6** Extract MediaPipe keypoints from recorded clips
- [ ] **5.7** LoRA fine-tune on custom medical PSL dataset

### Phase 6: Contest MVP Assembly (Strategy 3)

- [ ] **6.1** Build the Patient+'Doctor pipeline: PSL sequence classification over 30-50 phrases using LSTM or lightweight Transformer
- [ ] **6.2** Build the Doctor+'Patient pipeline: integrate OpenAI Whisper (voice+'text) + fuzzy phrase matching
- [ ] **6.3** Create or source pre-animated avatar clips for the 50 medical phrases
- [ ] **6.4** Build the demo UI: webcam input for Patient side, microphone input for Doctor side
- [ ] **6.5** End-to-end integration test: live two-way demo on webcam
- [ ] **6.6** Prepare the contest pitch: "Modular Framework" angle

---

## Success Criteria

| Milestone | Metric | Target |
|---|---|---|
| **Phase 3 (ASL Proof)** | Training completes without OOM | o. |
| | Loss decreases over 5+ epochs | o. |
| | VRAM usage stays under 15GB on T4 | o. |
| **Phase 4 (How2Sign)** | BLEU-4 on How2Sign test set | > 5.0 (prototype baseline) |
| | Model generates coherent English sentences | Qualitative pass |
| **Phase 5 (PSL)** | PSL dictionary scrape yields 200+ sign videos | o. |
| | Visual Encoder adapts to PSL keypoints without catastrophic forgetting | Loss continues to decrease |
| **Phase 6 (MVP)** | Patient+'Doctor classification accuracy on 50 phrases | > 90% |
| | Doctor+'Patient Whisper transcription accuracy (English + Urdu) | > 85% |
| | Live two-way demo runs in real-time on webcam | o. |

---

## Key Resources

| Resource | Link |
|---|---|
| T5_for_SLT Codebase | [github.com/zeleznyt/T5_for_SLT](https://github.com/zeleznyt/T5_for_SLT) |
| LINDAT YouTube-ASL Keypoints | [hdl.handle.net/11234/1-5898](http://hdl.handle.net/11234/1-5898) |
| YouTube-ASL Captions (Google) | [github.com/google-research/.../youtube_asl](https://github.com/google-research/google-research/tree/master/youtube_asl) |
| Research Paper (Zelezny et al.) | [arxiv.org/abs/2507.01532](https://arxiv.org/abs/2507.01532) |
| mT5 Model (HuggingFace) | [huggingface.co/google/mt5-small](https://huggingface.co/google/mt5-small) |
| PSL Dictionary | [psl.org.pk](https://psl.org.pk) |
| Strategy 1 (ASL Foundation) | [strategy_1-YT-ASL-H2S.md](../plans/strategy_1-YT-ASL-H2S.md) |
| Strategy 2 (PSL Medical) | [strategy_2-PSL-Medical.md](../plans/strategy_2-PSL-Medical.md) |
| Strategy 3 (Contest MVP) | [strategy_3-MVP-Contest.md](../plans/strategy_3-MVP-Contest.md) |
| Dataset Strategy | [dataset_strategy.md](../dataset_strategy.md) |
| Fairness Strategy | Notion Page ?" "Data Fairness & Demographic Balancing" |