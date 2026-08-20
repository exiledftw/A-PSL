# Memory Log: 2026-08-20

## Current Context
- **Goal**: Train a model to translate sign language videos into English text.
- **Current Phase**: Implementation Planning & Dataset Strategy finalization.
- **Collaborator**: Coordinating with Rehan's agent via this shared repository (`exiledftw/A-PSL`).

## Tasks Completed Today
- Created initial `Khizer.md` context file and established daily memory logging rules.
- Drafted, refined, and pushed a comprehensive `plan.md` defining the 3-stage model pipeline (Video → Gloss → English) optimized for T4 GPUs using MediaPipe keypoints (75 landmarks) and Transformers.
- Modified `plan.md` to include a fallback mechanism: using the Gemini API to automatically generate missing gloss annotations from English text in How2Sign.
- Created and pushed `dataset_strategy.md` outlining a dual-dataset approach:
  - **Primary**: How2Sign (for its clean gloss annotations and structured setup).
  - **Secondary (Augmentation)**: YouTube-ASL (for real-world diversity).
- Finalized a brilliant **"Streaming Storage Hack"** for YouTube-ASL: utilizing the pre-extracted LINDAT JSON keypoints (in 10 zip files) and mounting them directly into Kaggle via the "Remote Files" UI to completely bypass local 20GB storage limits.
- Stored the user's Notion API key securely in a local `scratch/notion_config.json` file.
- Used the Notion API to beautifully populate the user's "Datasets" page with the complete Dataset Strategy.
- Added a robust **"Data Fairness & Demographic Balancing"** section to the Notion page, acknowledging the lack of demographic metadata and inherent biases (especially regarding darker skin tones) in the YouTube-ASL dataset, and designing a 3-layer balancing strategy (ITA Skin Tone Estimation, Weighted Random Sampling, Stratified Validation) to mitigate it during training.

## Notes & Next Steps
- The dataset strategy and implementation blueprints are fully locked in and documented both on GitHub and Notion.
- Waiting to sync with Rehan's agent to begin executing Phase 1 (Data Acquisition & Preprocessing pipelines).