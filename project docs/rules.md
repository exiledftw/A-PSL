# Rules & Standards
# SANA Sign — Coding, Pipeline, and Avatar Rules

> **Project:** SANA Sign (A-PSL)
> **Purpose:** This document is the single source of truth for how to make changes to this codebase.
> **Rule of thumb:** If in doubt, read this file before touching anything.

---

## 1. The Golden Rules (Never Break These)

### R1 — The Pipeline Contract Is Sacred
The live inference script (`webcam_inference.py`) MUST match `keypoints.ipynb` byte-for-byte.
Any change to one MUST be reflected in the other. The specific things that must match:

| Setting | Current Value | Where It Lives |
|---|---|---|
| INPUT_DIM | 208 | Both files |
| TARGET_FRAMES | 60 | keypoints.ipynb + webcam_inference.py |
| MAX_SEQ_LEN | 100 | MedicalDataset + webcam_inference.py |
| Landmark order | [66 pose, 42 L-hand, 42 R-hand, 58 zeros] | Both files |
| Smoothing alpha | 0.75 (exponential) | Both files |
| API | MediaPipe Tasks API (NOT legacy mp.solutions) | Both files |

> VIOLATION = Silent model degradation. No error message. Predictions become garbage.

---

### R2 — Never Use WhatsApp to Transfer Videos
Video files MUST be transferred via USB cable or Google Drive. WhatsApp compresses video and destroys the spatial detail MediaPipe needs. This is documented in DATASET_RECORDING_GUIDELINES.md.

---

### R3 — Always Include fixDuplicateSkeleton() in Three.js
Y Bot.fbx has two overlapping skeletons: `Alpha_Joints` (outer) and `Alpha_Surface` (inner, used for mesh skinning). Without fixDuplicateSkeleton(), animations play internally but the visible mesh stays frozen in T-pose.

```javascript
// MANDATORY — call this after FBXLoader loads Y Bot.fbx
fixDuplicateSkeleton(fbxObject);
```

---

### R4 — BVH Retargeting Formula
Never inject BVH quaternions directly. Always compose with rest pose:

```javascript
// CORRECT
finalQuat = bvhQuat.multiply(restQuat);

// WRONG — breaks arm directions
finalQuat = bvhQuat;
```

Rest quaternions must be captured from the loaded Y Bot.fbx at scene load time.

---

### R5 — Confidence Threshold is 85%
The model must suppress output if confidence < 85% and display:
> "Low Confidence: Human Interpreter Required"

This is a safety requirement for SIMPACT. Do not lower this threshold without sign-off.

---

## 2. FBX Animation Rules

### R6 — Strip mixamorig: Prefix When Playing FBX in Three.js
FBXLoader returns animation tracks named `mixamorig:LeftArm.quaternion`.
Y Bot.fbx bones are named `LeftArm` (after fixDuplicateSkeleton strips the prefix).

```javascript
// MANDATORY in playFBX()
clip.tracks.forEach(t => {
    t.name = t.name.replace('mixamorig:', '');
});
```

Without this: AnimationMixer cannot bind tracks to bones → animation plays but character is stuck.

---

### R7 — FREEZE_BONES Only Contains Lower Body
Original code froze the left arm to avoid interference. This broke two-handed signs.
Current rule: only freeze legs and feet.

```javascript
const FREEZE_BONES = [
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
];
// DO NOT freeze LeftShoulder, LeftArm, LeftForeArm, LeftHand
```

---

### R8 — Hip Root Lock
The avatar drifts if BVH Hips X/Z translation is not locked.

```javascript
// In smoothQuaternionTrack / BVH application:
if (boneName === 'Hips') {
    track_x = restPosition.x;       // Lock X
    track_z = restPosition.z;       // Lock Z
    track_y = restPosition.y + (bvhY * 0.15); // Dampen Y by 85%
}
```

---

### R9 — BVH Spike Rejection
Quaternion spikes > 18 degrees between frames must be slerp-bridged, not applied directly.

```javascript
// In smoothQuaternionTrack:
const angleDiff = qPrev.angleTo(qCurrent);
if (angleDiff > Math.PI / 10) { // 18 degrees
    qCurrent = qPrev.slerp(qCurrent, 0.5); // bridge the spike
}
```

---

## 3. MediaPipe Extraction Rules

### R10 — Use Tasks API, Not Legacy API
```python
# CORRECT
from mediapipe.tasks.python import vision
HandLandmarker.create_from_options(options)

# WRONG — different output shape and ordering
import mediapipe as mp
mp.solutions.holistic.Holistic()
```

---

### R11 — Mirror Correction Default = ON
Training videos were recorded on iPhones with the front camera. MediaPipe swaps left/right in this setup. The MIRROR_CORRECTION flag defaults to True in webcam_inference.py. Always verify by checking on-screen "Detected: Left/Right" labels.

---

### R12 — Reset Tracker Per Clip
The exponential smoothing state must reset at the start of every new sign recording. Never let smoothing bleed between two separate signs.

```python
def reset_tracker():
    global prev_frame
    prev_frame = None  # Reset smoothing state
```

---

## 4. Dataset Recording Rules

### R13 — iPhone Settings
- Format: Most Compatible (NOT High Efficiency / HEIF)
- Resolution: 1080p @ 30fps ONLY (no 4K, no 60fps)
- Orientation: Landscape (horizontal)
- AE/AF Lock: Tap and hold on actor's chest before recording

### R14 — Return to Zero Rule
Every video MUST start and end with arms at sides (neutral pose), with 1 second pause.

```
[Arms down 1s] → [Perform sign] → [Arms down 1s] → [Stop recording]
```

Violating this causes avatar animation blending glitches at clip boundaries.

---

## 5. Avatar / VRM Rules

### R15 — Export SANA as VRM 0.0
VRM 1.0 is not yet supported by all Three.js VRM loaders or older Blender importers.
Always export as VRM 0.0 from VRoid Studio.

---

### R16 — Spine Damping for Torso Stability
When applying BVH to avatar, spine/neck/head tracks must be damped:

```javascript
const SPINE_DAMPING = 0.65; // 65% reduction
// Apply to: Spine, Spine1, Spine2, Neck, Head
```

Without this: torso wobbles excessively during arm signs.

---

## 6. Model Training Rules

### R17 — Collapse Diagnostic Before Trusting Any Prediction
Run the collapse check (d key) before declaring the model working:
- Real input → unique output ✓
- All-zeros → "Test are cheap here" ✓ (known fallback)
- Random noise → incoherent sentinel output ✓

All three must differ. If any two are identical, assume model collapse.

---

### R18 — Two-Step Temporal Processing (Never Skip Step 1)
```python
# STEP 1 — MANDATORY
frames_60 = resample_sequence(raw_frames, target=60)

# STEP 2 — MANDATORY
frames_100 = zero_pad(frames_60, max_len=100)

# NEVER do this — wrong temporal position for 1D Conv
frames_100 = resample_sequence(raw_frames, target=100)
```

---

## 7. Code Style Rules

### R19 — Print Statements: No Emoji on Windows
Windows console (cp1252) crashes on emoji in print statements.

```python
# WRONG
print("Extraction complete [emoji]")

# CORRECT
print("Extraction complete [DONE]")
```

---

### R20 — File Naming Convention
| Type | Convention | Example |
|---|---|---|
| FBX animations | snake_case phrase name | how_are_you_doing.fbx |
| BVH files | snake_case phrase name | do_you_feel_dizzy.bvh |
| Pose JSON | snake_case phrase name | how_are_you_doing.json |
| Model checkpoints | descriptive + size hint | sana_psl_complete_55mb.pt |
| Memory logs | role_YYYY-MM-DD_memory.md | khizer_2026-09-02_memory.md |

---

## 8. What Not to Do (Lessons Learned)

| Anti-Pattern | Why It Was Tried | Why It Was Abandoned |
|---|---|---|
| FBX file cleaning in Blender (manual) | Trying to clean jitter curves | Too slow; "waste of time" per user |
| Using Rokoko for live capture | Alternative to XR Animator | "Rokoko sucks" |
| Kalidoface for live avatar | Web-based webcam avatar | Too jittery |
| Freezing LeftArm in FREEZE_BONES | Prevent drift | Broke two-handed signs (paani.fbx signs left) |
| Resampling directly to 100 frames | Simpler code | Silently misaligns 1D Conv temporal learning |
| Using legacy MediaPipe Holistic | Familiar API | Different output shape from training pipeline |
| VRM 1.0 export | Newer standard | Compatibility issues with web players |
