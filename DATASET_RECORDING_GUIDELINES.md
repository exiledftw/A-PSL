# SIMPACT 2026: Official Dataset Recording Guidelines

This document outlines the strict rules for recording the Sign Language dataset using iPhones. Because our AI relies on mathematical 3D coordinate extraction (MediaPipe), failing to follow these rules will corrupt the dataset and break the 3D Avatar animations.

## 📱 1. iPhone Camera Setup (CRITICAL)
Before hitting record, the iPhone owner MUST change these settings:
*   **Format:** Go to Settings > Camera > Formats > Select **"Most Compatible"**. (If you use "High Efficiency", Python OpenCV will crash on Windows).
*   **HDR Video:** Go to Settings > Camera > Record Video > Turn **OFF** HDR Video.
*   **Resolution:** Set strictly to **1080p at 30 FPS**. (Do not use 4K or 60 FPS).
*   **Orientation:** The phone MUST be completely horizontal (Landscape / 16:9). 
*   **AE/AF Lock:** Before recording, tap and hold on the actor's chest on the screen until the yellow **"AE/AF LOCK"** badge appears. This stops the camera from blurring out of focus when hands move forward.

## 🎬 2. Framing & Environment
*   **The Background:** Stand against a completely plain, solid-colored wall (preferably light grey or white). Do not record with doors, windows, plants, or other people in the background.
*   **The Framing:** Frame the shot from the belly-button to 6 inches above the head. 
*   **The Boundary Rule:** The actor's hands MUST stay inside the camera screen at all times. If a hand swings out of frame, the AI immediately outputs `0,0,0` and ruins the sequence.

## 👕 3. Wardrobe
*   **Short Sleeves Only:** Actors must wear short sleeves (like polo shirts or t-shirts). MediaPipe relies on bare wrists and elbows for accurate tracking.
*   **Solid Colors:** Wear a solid, dark color (black, navy, dark blue) that contrasts heavily against the light-colored wall. 

## 🧍 4. The "Return to Zero" Rule (For the Avatar)
To ensure the 3D Avatar can blend smoothly between different animations without glitching, EVERY single video must follow this exact sequence:
1.  Stand with arms resting naturally at your sides. Wait 1 second.
2.  Perform the sign clearly and at a moderate pace.
3.  **Drop your arms completely back to your sides (Neutral Pose).** Wait 1 second.
4.  *Now* stop recording. 

---

## 🚫 THE "DO NOT DO" LIST
*   **DON'T** record vertically (Portrait mode). This breaks the spatial mathematics.
*   **DON'T** stop the recording while your hands are still raised in the air.
*   **DON'T** wear baggy hoodies, long sleeves, or clothing with busy patterns/logos.
*   **DON'T** record outside in a garden or under harsh dappled sunlight (shadows confuse the AI).
*   **DON'T** sign faster than normal. Fast motion creates camera blur, causing the AI to lose the fingers.
*   **DON'T** transfer the final videos via WhatsApp. WhatsApp heavily compresses the files and ruins the quality. Use a USB cable or Google Drive to transfer the `.MP4` files to the PC.
