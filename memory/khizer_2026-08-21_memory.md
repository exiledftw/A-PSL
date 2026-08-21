# Memory Log: 2026-08-21

## Current Context
- **Goal**: Build a two-way Medical Pakistani Sign Language (PSL) MVP.
- **Current Phase**: Phase 1 - Data Logistics (Kaggle Pipeline Setup).
- **Collaborator**: Syncing with Rehan's agent via the unified `docs/Project_Whitepaper.md` architecture.

## Tasks Completed Today
- Read and synthesized Rehan's agent's memory and the Strategy 2 & 3 documents.
- Deprecated `plan.md` and officially transitioned the project's single source of truth to `docs/Project_Whitepaper.md`.
- Initiated **Task 1.1 & 1.2**: User successfully linked the massive 34GB LINDAT keypoint dataset (`raw_file1.zip`) into Kaggle via the Remote Files UI. It is currently unzipping in the background.
- **Critical Data Pivot**: Discovered that the LINDAT dataset includes a bundled `YT.translations.all.json` (56MB) file. We will use this natively nested dictionary (O(1) lookup, pre-formatted `video_id.start-end` keys) instead of writing a complex Pandas joiner for the Google Research TSV. 
- Updated `checklist/item_1.md` to reflect the removal of the Google TSV dependency and the superiority of the LINDAT JSON translations.
- **Critical Kaggle Optimization:** Kaggle's backend failed to unzip the LINDAT dataset because the API URL lacks a `.zip` extension (saving it as a 34GB blob named `content`). Instead of renaming or re-downloading, established a zero-storage footprint workaround using Python's `zipfile` library to stream JSON files directly out of the `content` blob on the fly. This prevents Kaggle's `/kaggle/working` directory from crashing (20GB limit).
- Provided the user with the foundational PyTorch Dataset class (`YouTubeASLDataset`) and explained how the nested dictionary keys (e.g. `BSRkKugmny0.006331-006508.json`) perfectly map to the individual files inside the zip without needing any pandas dataframe logic.
- **Kaggle Validation & Fix:** Fixed a file path issue in the zipfile extraction script for Kaggle (`/kaggle/input/datasets/kkmalik/yt-asl/content`), verifying the user can pull 100 demo keypoint files (`yt_demo_vids.zip`) successfully.
- **Documentation Sync:** Overcame local push limits by breaking down the updates. Successfully pushed the exact architecture and Kaggle storage bypass updates into `docs/Project_Whitepaper.md` and `checklist/item_1.md`.
- **Model Generation Blueprint:** Gathered full repository context from the `zeleznyt/T5_for_SLT` codebase (including custom normalizations and linear projections) to author a highly robust, non-negotiable prompt blueprint for Claude Sonnet. This prompt will force Claude to generate the ultimate, compliant 11-cell `.ipynb` Kaggle training loop.
- **Architecture Visualization:** Created and pushed `docs/main_diagram.md` showcasing a visual Mermaid flowchart of the Hybrid Architecture (Input -> Eyes -> Bridge -> Brain -> Output) to solidify understanding.
- **Conceptual Grounding:** Explained the core intuition of the Cross-Modal AI (Spatial-Temporal Transformer + Frozen mT5) to the user, drawing the analogy of a "Charades game" to demystify backpropagation and the purely mathematical (non-textual) nature of the 208 MediaPipe coordinates.

## Notes & Next Steps
- Waiting for the user to execute the Claude prompt and run the generated Kaggle notebook.
- Monitor Kaggle VRAM usage on the first dummy batch.