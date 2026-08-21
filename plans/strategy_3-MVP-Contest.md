# SANA Sign - SIMPACT 2026 Prototype

**Showcase Date:** September 17, 2026 (SIMPACT 2026, CIME Karachi)
**Scope:** A PROTOTYPE only - not a clinically validated or fully deployed product. All strategies below are scoped to this constraint.

For a contest prototype (Minimum Viable Product), a massive 3-phase LLM training pipeline is overkill and too slow. Instead, the focus should be on a **Highly Constrained Vocabulary** and **End-to-End Visual Flow**. 

We will build two pipelines under the SANA Sign module (part of the SANA AI HIMS):
1. **Patient to Doctor:** PSL Gestures → English/Urdu Text
2. **Doctor to Patient:** Voice → Text → Avatar PSL Gestures

---

## 1. PSL to Text (Patient → Doctor)

Instead of continuous, open-ended translation (which requires thousands of hours of data and T5 models), the MVP should use **Sequence Classification** over a fixed set of medical phrases.

### The MVP Approach
*   **Vocabulary:** Define 30-50 critical medical phrases (e.g., "I have a headache," "Where is the pain?", "I am dizzy," "Take this medicine").
*   **Data Collection (1 Weekend):** Have 2-3 people record themselves signing these 50 phrases (maybe 10-15 variations each). 
*   **Preprocessing:** Extract MediaPipe keypoints (Face, Hands, Pose) from these videos.
*   **Model:** A lightweight **LSTM**, **GRU**, or **Action Recognition Transformer**. 
*   **Why it wins contests:** It trains in minutes on a laptop, hits 95%+ accuracy because the vocabulary is constrained, and runs in real-time on a webcam without needing a heavy GPU.

---

## 2. Voice to Avatar (Doctor → Patient)

Training a generative AI to create dynamic 3D avatar movements from scratch is a massive research problem. For an MVP, we use a **Dictionary-Based Blending Approach**.

### The MVP Pipeline
1.  **Voice to Text:** 
    *   Use an off-the-shelf API like **OpenAI Whisper** or **Google Cloud Speech-to-Text**. Whisper is excellent at handling both English and Urdu.
2.  **Text to Gloss (Simplified):** 
    *   For the MVP's 50 phrases, map the incoming text directly to a known phrase ID using simple NLP similarity (e.g., fuzzy string matching or a lightweight BERT embedding).
3.  **Avatar Animation (The "Magic" Trick):**
    *   **Pre-animate:** Create or use pre-recorded 3D animations (using tools like Unity, Unreal, or even just 2D video clips of an avatar) for the 50 phrases or root words.
    *   **Trigger & Blend:** When the doctor says "Take this medicine," Whisper transcribes it, the system matches it to the phrase ID, and the Unity/Web interface immediately triggers the corresponding pre-made 3D animation. 
    *   **Why it wins contests:** It looks incredibly polished, works instantly, and completely avoids the uncanny valley of AI-generated skeleton movements. 

---

## 3. Safety Framework & Fallback Mechanism (SIMPACT Requirement)

To ensure clinical safety during prototype demonstrations, the system is designed with a strict confidence threshold. 
*   **Confidence Thresholds:** During Patient → Doctor sign classification, the model outputs a confidence percentage (e.g., 92%).
*   **Human Escalation:** If the confidence falls below a clinically validated threshold (e.g., 85%), the system will automatically suppress the prediction and display an escalation message to the clinician: *"Low Confidence: Human Interpreter Required."* This prevents dangerous medical mistranslations from being shown to the doctor.

---

## 4. Non-Technical SIMPACT Deliverables

To fully satisfy the SIMPACT reviewer feedback, the following documentation and protocols must be drafted alongside the technical build:
*   **PSL Dataset Details:** Full documentation of our custom 50-phrase medical dataset (demographics, recording conditions).
*   **Video Recording Consent Process:** A drafted UI/UX flow and formal consent form for capturing patient video in a clinical setting.
*   **Clinical Validation Study Design:** A written plan outlining how SANA Sign *will* be tested in a formal clinical trial post-prototype.
*   **Avatar Acceptance Testing:** Documented feedback sessions with deaf community members assessing the clarity of our 3D avatar animations.

---

## 5. The Contest Presentation Angle
You pitch this not as a limited system, but as a **Modular Framework**. 
*   "Today, we are demonstrating the core engine with 50 critical emergency room phrases."
*   "Our architecture is designed to scale. As we collect more data, the classification engine swaps out for a T5-Transformer, and our avatar dictionary swaps out for a Generative motion model."

This proves you have a working product *today* while showing you understand the roadmap for tomorrow.