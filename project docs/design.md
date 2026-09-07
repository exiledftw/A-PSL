# Design Document
# SANA Sign — UI/UX, Avatar, and System Design

> **Project:** SANA Sign (A-PSL)
> **Purpose:** Covers visual design, avatar aesthetics, interaction design, and UX decisions.

---

## 1. Design Philosophy

SANA Sign must be usable by two very different audiences at the same time:

1. **Deaf Patient** — may have limited Urdu/English literacy, unfamiliar with tech, is in distress
2. **Doctor** — time-constrained, expects information clearly and fast

Design decisions must serve both simultaneously. The screen is split: patient side (avatar + output) on one half, doctor side (text translation + controls) on the other.

### Design Principles
- **Clarity over cleverness** — every UI element must be understandable in 3 seconds
- **Visual-first for deaf user** — patient sees avatar, signs, and visual feedback (not text)
- **Minimum friction for doctor** — one button starts recording, output appears automatically
- **Safety is visible** — low-confidence warnings must be impossible to miss (red, large font)
- **Offline-capable** — no cloud dependency for demo day

---

## 2. Screen Layout Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SANA SIGN                                    │
├─────────────────────────┬───────────────────────────────────────────┤
│                         │                                           │
│   SANA AVATAR           │   DOCTOR PANEL                           │
│   (Three.js WebGL)      │                                           │
│                         │   Patient said:                           │
│   [3D Nurse Animation]  │   ┌─────────────────────────────────┐   │
│   Performs PSL sign     │   │  "I have a severe headache"     │   │
│   matching doctor's     │   │  "مجھے شدید سر درد ہے"         │   │
│   selected phrase       │   └─────────────────────────────────┘   │
│                         │                                           │
│                         │   Confidence: 94%  [██████████░] check   │
│                         │                                           │
│                         │   Doctor speaks:                          │
│                         │   [Mic: Press to Speak]                  │
│                         │                                           │
│                         │   Or select a phrase:                    │
│                         │   [Where is the pain?    dropdown]       │
│                         │                                           │
├─────────────────────────┴───────────────────────────────────────────┤
│   PATIENT: [Record Sign]      Status: Listening...                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Avatar — SANA

### 3.1 Character Design
| Attribute | Detail |
|---|---|
| **Name** | SANA |
| **Role** | Medical nurse / interpreter avatar |
| **Creator** | Reyhan |
| **Tool** | VRoid Studio 2.14 |
| **Version** | 1.0.0 |
| **Copyright** | 2026 Bunby |
| **Polygon count** | ~33,343 |
| **Material count** | 14 |
| **Export format** | VRM 0.0 |
| **Clothing** | Nurse uniform |

### 3.2 Why VRM 0.0 (Not VRM 1.0)
VRM 0.0 was chosen because:
- Supported by Three.js VRM loaders without extra plugins
- Works with XR Animator, Warudo, VSeeFace, and older Blender add-ons
- VRM 1.0 has breaking changes in bone naming + blend shapes not yet universally supported

### 3.3 Why VRoid (Not Commercial Avatar)
- Customizable: can match PSL signer body proportions
- Free, no licensing issues for SIMPACT
- VRM format = plug-and-play across tools
- Community avatars on VRoid Hub and BOOTH (free nurse models available)

### 3.4 Avatar Rendering Stack
```
SANA.vrm (character data)
    │
    ▼
Y Bot.fbx (Mixamo humanoid — provides the skeleton Three.js drives)
    │
    ├── FBX animation tracks (pre-baked PSL signs)
    └── BVH retargeted animations (DeepMotion captures)
         │
         ▼
Three.js WebGL (avatar_player.html)
    │
    ▼
Browser display (no install)
```

### 3.5 Known Avatar Issues and Fixes

| Issue | Cause | Fix |
|---|---|---|
| Head tracking feels "soft" (floaty) | Head/Neck Smoothing in XR Animator too high | Lower Head/Neck smoothing slider to < 30 |
| Salute gesture needs hand far from head | VRoid default head proportionally large vs real human | Reduce Head Size slider in VRoid body settings |
| Character drifts during gestures | BVH Hip translation uncontrolled | Hip X/Z locked; Y dampened 85% (rules.md R8) |
| Torso wobbles during arm signs | Spine tracks too reactive | 65% spine damping applied (rules.md R16) |
| Arms frozen in T-pose during some BVH | FREEZE_BONES included left arm | Fixed: FREEZE_BONES = lower body only (rules.md R7) |

---

## 4. Interaction Design

### 4.1 Patient Flow (Gesture-to-Text)

```
Patient sits facing webcam
         │
         ▼
Patient sees: [Record Sign button]
         │
         ▼
Patient performs sign (system records)
         │
         ▼
Patient presses button again (or auto-detects stop)
         │
         ▼
System displays:
  - English translation (large font, for doctor)
  - Urdu translation (for bilingual reference)
  - Confidence meter (visible to doctor)
         │
    ┌────┴────┐
    │         │
  >= 85%    < 85%
    │         │
 Show text   Show:
             "Low Confidence
              Human Interpreter Required"
             (red banner, large)
```

### 4.2 Doctor Flow (Text-to-Avatar)

```
Doctor wants to communicate something to deaf patient
         │
         ├── OPTION A: Speak
         │   Doctor presses [Mic Speak] button
         │   Whisper STT transcribes in real-time
         │   NLP matches to phrase ID
         │   SANA avatar performs matching PSL sign
         │
         └── OPTION B: Select Phrase
             Doctor opens dropdown: 40 medical phrases
             Selects one (e.g. "Where is the pain?")
             SANA avatar performs matching PSL sign
```

### 4.3 Keyboard Controls (Dev Mode)
| Key | Action |
|---|---|
| SPACE | Start/stop sign recording |
| m | Toggle mirror correction (L/R hand flip) |
| d | Run collapse diagnostic |
| q | Quit |

---

## 5. Color and Visual Language

| Element | Design Choice | Reason |
|---|---|---|
| Avatar background | Soft blue-grey gradient | Medical, calm, non-distracting |
| Confidence bar (high) | Green | Universal "good" signal |
| Confidence bar (low) | Red + pulsing | Impossible to miss; safety-critical |
| Translation text | Large (28pt+), high contrast | Doctor reads quickly at desk distance |
| Urdu text | Right-aligned, RTL font (Noto Nastaliq Urdu) | Correct Urdu typography |
| SANA avatar lighting | Three-point lighting, soft shadows | Professional medical appearance |
| Button style | Large rounded rectangle, flat color | Touch-friendly for both user types |

---

## 6. The Two Live Capture Approaches

### For SIMPACT Demo (No Internet Needed)
- avatar_player.html — serves animations from local files
- Pre-baked FBX/BVH animations for all 40 phrases
- No webcam needed for doctor side (use dropdown instead)
- Patient side webcam inference runs locally via Python

### For Post-Demo Live Mode (Internet OK)
- XR Animator (web) or Warudo (Steam) for live webcam → SANA avatar
- Doctor speaks → Whisper API → phrase match → local FBX trigger
- Patient webcam → Python inference → translation display

---

## 7. Accessibility & Localization

| Feature | Status |
|---|---|
| Urdu script display (RTL) | Required — deaf patients may read Urdu |
| English translation | Required — for doctor |
| High-contrast UI mode | Recommended for bright hospital lighting |
| Large text mode (28pt+) | Required — readability at arm's length |
| Sound cues | NOT used — deaf patient; use visual-only feedback |
| PSL sign clarity | Avatar must be visible from 1–2 meters |

---

## 8. Motion Design for PSL Signs

### 8.1 What Makes a Good PSL Avatar Sign
- **Start and end at neutral pose** — arms at sides, slight natural bend
- **Sign speed: 70–90% of human speed** — slightly slower for clarity
- **Finger detail preserved** — critical for many PSL signs (use DeepMotion capture for fingers)
- **No torso wobble** — patient focus should be on hands and arms only
- **Consistent framing** — avatar fills 60–80% of screen height

### 8.2 Animation Sources by Priority
| Priority | Source | Quality | Finger Capture |
|---|---|---|---|
| 1st | Khizer DeepMotion FBX (cleaned) | High | Yes (400 curves) |
| 2nd | Custom MediaPipe → FBX pipeline | Medium | No fingers |
| 3rd | BVH from DeepMotion (with smoothing) | Medium | Partial |
| Fallback | Still image of hand shape | Low | N/A |

### 8.3 Animation Blending
Between each sign, the avatar must return to neutral pose. This is enforced by:
1. DATASET_RECORDING_GUIDELINES.md "Return to Zero" rule (for recorded animations)
2. Crossfade duration: 0.3s slerp between clip end and neutral pose start

---

## 9. File Organization for Design Assets

```
exiledftw/A-PSL
│
├── Y Bot.fbx                   ← Skeleton for Three.js
├── avatar_player.html           ← Main player UI
│
├── medical_dataset/             ← FBX animation clips
│   ├── how_are_you_doing.fbx
│   └── how_do_you_feel_clean.fbx
│
└── bvh_files/                   ← BVH animation clips
    ├── I_was_born_deaf.bvh
    ├── do_you_feel_dizzy.bvh
    ├── how_do_you_feel.bvh
    └── keep_your_feet_moisturized.bvh
```

---

## 10. SIMPACT Presentation Design

The 10-minute demo should follow this script:

| Time | Scene | What Happens |
|---|---|---|
| 0:00–1:00 | Problem | Show stat (1M deaf Pakistanis, no PSL interpreters) |
| 1:00–3:00 | Patient→Doctor | Demo: signer signs a phrase, model translates it on screen |
| 3:00–4:00 | Safety | Show confidence gate: bad sign → red warning |
| 4:00–6:00 | Doctor→Patient | Demo: speak phrase → SANA avatar signs it |
| 6:00–7:00 | Architecture | Brief system diagram (this doc, architecture.md) |
| 7:00–9:00 | Roadmap | "Today: 40 phrases. Tomorrow: full clinical vocabulary" |
| 9:00–10:00 | Q&A | Answer with confidence threshold and clinical plan |

### Pitch Frame
> "Today we demonstrate the core engine with 40 critical emergency-room phrases.
> Our architecture is modular — as we collect more data, the classifier scales to a T5 Transformer,
> and our animation dictionary scales to generative motion synthesis."
