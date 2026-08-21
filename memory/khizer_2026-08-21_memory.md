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

## Notes & Next Steps
- Tested the `zipfile` reading approach which bypasses the need for Kaggle to unzip the payload.
- Finalize the PyTorch `Dataset` script to load the JSON skeletons using the `YT.translations.all.json` keys into the Model.