# Strategy 2: PSL Medical Translation (Doctor-Patient)

## 1. The Challenge: Data Scarcity
Unlike American Sign Language (ASL), which has massive continuous datasets (YouTube-ASL, How2Sign), **Pakistani Sign Language (PSL)** suffers from extreme data scarcity. Public datasets (like PakSign, UAlpha40, or the FESF dictionary) are limited to isolated words or alphabets. There is no existing public dataset of continuous PSL medical conversations. 

Therefore, our strategy must rely on **Transfer Learning** and a **Custom Data Collection Pipeline**.

## 2. Architecture Updates (The "Same Approach")
We will keep the highly efficient, compute-friendly T4 architecture:
*   **Input:** MediaPipe 2D Keypoints (bypassing raw video storage).
*   **Visual Encoder:** Spatial-Temporal Transformer.
*   **Projection Layer:** Linear mapping to LLM latent space.
*   **Decoder (Updated):** Instead of standard T5 (English only), we should use **mT5 (Multilingual T5)** if the desired output text is in Urdu, or keep **T5** if the doctors prefer English translations of the PSL. 

## 3. The 3-Phase Training Strategy

### Phase 1: Foundation Pre-training (ASL Transfer Learning)
*   **Dataset:** YouTube-ASL (600k clips) & How2Sign.
*   **Action:** Train the Custom Visual Encoder while keeping the T5/mT5 Decoder frozen.
*   **Why?** Sign languages share physical characteristics (hand tracking, facial expressions, spatial grammar). By pre-training on massive ASL data, the Visual Encoder learns "how to see" sign language generally, preventing the model from starting from scratch on the much smaller PSL data.

### Phase 2: PSL Domain Adaptation (General PSL)
*   **Dataset:** Public PSL resources (FESF PSL Dictionary scraped via open-source scripts, PakSign dataset).
*   **Action:** Continue training the Visual Encoder on this data.
*   **Why?** This shifts the model's understanding from ASL phonology to PSL phonology (e.g., specific hand shapes and movements unique to PSL).

### Phase 3: Medical Fine-Tuning (Custom Dataset + LoRA)
*   **Dataset:** A custom-recorded dataset of PSL medical dialogues (Doctor-Patient interactions). 
*   **Action:** 
    1. Freeze the base T5/mT5 Decoder.
    2. Inject **LoRA (Low-Rank Adaptation)** adapters into the Decoder.
    3. Unfreeze the Visual Encoder.
    4. Train the Visual Encoder and LoRA adapters jointly on the custom medical dataset.
*   **Why?** LoRA allows us to adapt the powerful language model to the specific medical jargon and conversational flow of doctor-patient interactions without causing catastrophic forgetting or exceeding the 15GB VRAM limit of a T4 GPU.

## 4. Custom Medical Dataset Blueprint
Since we must record this data, here is the protocol:
1.  **Scripting:** Draft 500-1,000 common doctor-patient scenarios (e.g., describing symptoms, prescribing medication, asking about pain levels).
2.  **Recording:** Hire fluent PSL interpreters to sign both the doctor and patient sides.
3.  **Extraction:** Run MediaPipe locally or on cloud VMs to extract the 75 keypoints (Face, Pose, Hands) and save them as lightweight JSON/NPY files.
4.  **Annotation:** Pair the keypoint sequences with their text translations (English or Urdu).

## 5. Summary
By leveraging the existing VLM-style pipeline (Visual Encoder + LoRA Decoder) but adding a crucial **Transfer Learning** step from ASL to PSL, your company can achieve state-of-the-art results without needing millions of PSL videos. You will only need to invest in a small, high-quality custom medical dataset for Phase 3.