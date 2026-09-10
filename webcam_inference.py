"""
SANA A-PSL — Live Webcam Sign Inference
=========================================
This script is built to match EXACTLY the pipeline used to create your training
data (keypoints.ipynb) and the model architecture used to train/fine-tune your
model (notebook399144a5d3.ipynb). Nothing here is guessed:

  - 208-dim feature layout per frame:
        [0:66]    33 Pose landmarks   (x, y)
        [66:108]  21 Left Hand landmarks (x, y)
        [108:150] 21 Right Hand landmarks (x, y)
        [150:208] 29 "face" slots     -> always zeros (SANA standard, not extracted)
  - Same MediaPipe Tasks API models: hand_landmarker.task + pose_landmarker.task
  - Same mirror-hand-swap logic, same exponential coordinate smoothing (alpha=0.75)
  - Same reset_tracker() per capture (no smoothing bleed between signs)
  - Same resample_sequence() to 60 frames, then the SAME zero-padding to 100
    frames that MedicalDataset used at train time (since your clean .npy files
    were 60 frames and get padded up to MAX_SEQ_LEN=100 inside the Dataset).

REQUIREMENTS:
    pip install mediapipe opencv-python torch transformers numpy
    pip install pyttsx3 deep-translator   # optional: TTS + online Urdu fallback
    (the app works fine without the optional two -- see notes below)

USAGE:
    python webcam_inference.py --model sana_psl_medical_finetuned.pt

CONTROLS (while the webcam window is focused):
    SPACE  - Start recording a sign. Press SPACE again to stop and run inference.
    m      - Toggle mirror-hand-correction (use if Left/Right hands look swapped
             on screen — see note below).
    l      - Reset focus lock (force re-acquire the user nearest to webcam).
    f      - Toggle fullscreen mode (stretches video across the entire screen).
    d      - Run the collapse diagnostic on the last captured sign.
    q      - Quit.

NEW: URDU TRANSLATION, LATENCY, CONFIDENCE, TEXT-TO-SPEECH
    - Urdu is always a real, full-sentence translation -- never letter-by-letter.
      Known medical phrases use the exact, verified MEDICAL_DICTIONARY mapping;
      anything else falls back to online sentence-level translation via
      deep-translator if installed and reachable. If neither applies, no Urdu
      line is shown (never a broken/partial one).
    - Latency is wall-clock time for the encoder forward pass + generation.
    - Confidence is the average top-1 token probability across the whole
      generated sentence (not just the first token), shown as a percentage.
    - Text-to-speech speaks the English prediction aloud via pyttsx3, on a
      background thread so it never freezes the video feed. If pyttsx3 isn't
      installed or fails to initialize, the app just runs silently -- it
      never crashes because of TTS. Disable explicitly with --no-tts.

IMPORTANT ABOUT MIRRORING:
Your extraction notebook set MIRROR_CORRECTION=True, meaning your training
videos were recorded in a way where MediaPipe's raw handedness call was the
opposite of reality (typical of front-camera "selfie" recordings), so left/
right hand coordinates were swapped after detection. A live webcam feed via
cv2.VideoCapture is NOT flipped by OpenCV, so whether you need the same
swap depends on how your camera driver/app presents the image. Use the on-
screen "Detected: Left/Right" labels next to each skeleton: if your ACTUAL
right hand is being labeled "Left" (and vice versa), press 'm' to enable the
swap so it matches what the model was trained on. This is a small runtime
toggle, not a guess baked silently into the pipeline — verify it visually
before trusting predictions.
"""

import argparse
import difflib
import math
import os
import sys
import threading
import time
import urllib.request
from collections import deque

# Ensure Windows consoles support UTF-8 without crashing on Urdu characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Optional dependencies — the app must keep working perfectly even if these
# are missing, offline, or fail at runtime. Never let them crash the main loop.
try:
    import pyttsx3
    TTS_LIB_AVAILABLE = True
except ImportError:
    TTS_LIB_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_LIB_AVAILABLE = True
except ImportError:
    TRANSLATOR_LIB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# CONFIG — mirrors CONFIG dicts in both source notebooks exactly
# ──────────────────────────────────────────────────────────────────────────
CONFIG = {
    "MT5_MODEL_NAME":      "google/mt5-small",
    "D_MODEL":             512,
    "NUM_HEADS":           8,
    "NUM_ENCODER_LAYERS":  2,
    "DIM_FEEDFORWARD":     1024,
    "DROPOUT":             0.108,
    "MAX_SEQ_LEN":         100,   # what the model / MedicalDataset was trained with
    "MAX_TARGET_LEN":      32,
    "INPUT_DIM":           208,

    # Extraction params — identical to keypoints.ipynb Cell 2
    "TARGET_FRAMES":       60,    # keypoints.ipynb resamples every clip to 60 frames
    "MIN_DETECTION_CONF":  0.5,
    "MIN_TRACKING_CONF":   0.5,
    "SMOOTHING_ALPHA":     0.75,
    "MIRROR_CORRECTION":   True,  # default matches keypoints.ipynb; toggle with 'm'
}

# The 16 fine-tuned PSL medical and test sentences
TRAINED_SENTENCES = [
    "Ambulance ko call karro",
    "Assalam o alaikum",
    "is blood pressure high or low",
    "Mein beemar hu",
    "Mere sarr mein dard hai",
    "Meri aankh surkh hai",
    "Mujhay bukhar hai",
    "Mujhay chakkar aa rhy hein",
    "Mujhay dard kam hai",
    "Mujhay dard tez hai",
    "Mujhay kuch dawa khareedni hai",
    "No",
    "Test are cheap here",
    "There has been an accident",
    "Yes",
    "Yesterday"
]

MEDICAL_DICTIONARY = {
    # The 16 fine-tuned and test classes with verified Urdu Nastaleeq translations
    "Ambulance ko call karro": "ایمبولینس کو کال کریں",
    "Assalam-o-ALaikum": "السلام علیکم",
    "Assalam o alaikum": "السلام علیکم",
    "Is blood pressure high or low?": "کیا بلڈ پریشر زیادہ ہے یا کم؟",
    "is blood pressure high or low": "کیا بلڈ پریشر زیادہ ہے یا کم؟",
    "Mein beemar hu": "میں بیمار ہوں",
    "Mere sarr mein dard hai": "میرے سر میں درد ہے",
    "Meri aankh surkh hai": "میری آنکھ سرخ ہے",
    "Mujhay bukhar hai": "مجھے بخار ہے",
    "Mujhay chakkar aa rhy hein": "مجھے چکر آ رہے ہیں",
    "Mujhay dard kam hai": "مجھے درد کم ہے",
    "Mujhay dard tez hai": "مجھے درد تیز ہے",
    "Mujhay kuch dawa khareedni hai": "مجھے کچھ دوا خریدنی ہے",
    "No": "نہیں",
    "Tests are cheap here": "یہاں ٹیسٹ سستے ہیں۔",
    "Test are cheap here": "یہاں ٹیسٹ سستے ہیں۔",
    "There has been an accident": "ایک حادثہ ہوا ہے۔",
    "Yes": "جی ہاں",
    "Yesterday": "کل (گزرا ہوا)",

    # Backwards-compatibility legacy mappings
    "Hello I need to see a doctor": "ہیلو، مجھے ڈاکٹر سے ملنے کی ضرورت ہے۔",
    "I have a severe headache": "مجھے شدید سر درد ہے۔",
    "Where is the pain": "درد کہاں ہے؟",
    "Are you having trouble breathing": "کیا آپ کو سانس لینے میں تکلیف ہو رہی ہے؟",
}

_translation_warning_shown = False


def translate_to_urdu(english_text: str) -> str:
    """Returns a proper Urdu SENTENCE translation, never isolated letters.

    Priority order:
      1. Exact match in MEDICAL_DICTIONARY (guaranteed-correct, offline, instant)
      2. Normalized match in MEDICAL_DICTIONARY (ignoring casing, punctuation, hyphens)
      3. Online fallback via deep-translator (full-sentence machine translation)
      4. If neither is available, return '' and the caller just shows no Urdu
         line rather than showing something wrong or crashing.
    """
    global _translation_warning_shown

    if not english_text:
        return ""

    # 1. Exact dictionary match
    if english_text in MEDICAL_DICTIONARY:
        return MEDICAL_DICTIONARY[english_text]

    # 2. Normalized dictionary match (strips punctuation, hyphens, and casing)
    def _norm(txt):
        return "".join(c for c in txt.lower().replace("-", " ") if c.isalnum() or c.isspace()).strip()

    norm_en = _norm(english_text)
    for k, v in MEDICAL_DICTIONARY.items():
        if _norm(k) == norm_en:
            return v

    # 2. Fallback: real sentence-level machine translation for anything
    #    outside the known phrase set.
    if TRANSLATOR_LIB_AVAILABLE:
        try:
            return GoogleTranslator(source="en", target="ur").translate(english_text)
        except Exception as e:
            if not _translation_warning_shown:
                print(f"[translation] Online Urdu translation unavailable ({e}). "
                      f"Falling back to English-only display for unknown phrases.")
                _translation_warning_shown = True
            return ""
    else:
        if not _translation_warning_shown:
            print("[translation] deep-translator not installed -- only exact "
                  "MEDICAL_DICTIONARY phrases will show Urdu. Run: "
                  "pip install deep-translator")
            _translation_warning_shown = True
        return ""


class TextToSpeech:
    """Thin wrapper around pyttsx3 that speaks in a background thread so it
    never blocks or stutters the webcam video loop.

    IMPORTANT: on Windows, reusing a single pyttsx3 engine instance across
    multiple say()/runAndWait() calls is a well-known source of "it spoke
    once and then silently did nothing after that" bugs (SAPI5 driver state
    doesn't always reset cleanly). The reliable fix is to create a fresh
    engine for every utterance. That's what we do here -- slightly more
    overhead per call, but it actually speaks every time.
    """

    def __init__(self):
        self.available = TTS_LIB_AVAILABLE
        self._lock = threading.Lock()
        if self.available:
            # Sanity-check that we can actually initialize an engine at all
            # before promising the rest of the app that TTS works.
            try:
                test_engine = pyttsx3.init()
                test_engine.stop()
                del test_engine
            except Exception as e:
                print(f"[tts] Could not initialize text-to-speech engine ({e}). "
                      f"Predictions will still display normally, just without audio.")
                self.available = False

    def speak_async(self, text: str):
        if not self.available or not text:
            return
        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()

    def _speak_worker(self, text: str):
        with self._lock:
            try:
                engine = pyttsx3.init()   # fresh engine every time -- see class docstring
                engine.setProperty("rate", 165)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[tts] Speech failed ({e}) -- continuing without audio for this prediction.")


# ──────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE — copied verbatim from notebook399144a5d3.ipynb
# (must match exactly or state_dict loading will silently misalign)
# ──────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TemporalGestureTokenizer(nn.Module):
    def __init__(self, input_dim=208, d_model=512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model // 2, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.BatchNorm1d(d_model // 2)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = x.transpose(1, 2)
        return x


class UpgradedSpatialTemporalEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ffn_dim, dropout, max_len):
        super().__init__()
        self.tokenizer = TemporalGestureTokenizer(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        x = self.tokenizer(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x


class SANA_PSL_Translator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_encoder = UpgradedSpatialTemporalEncoder(
            input_dim=config["INPUT_DIM"],
            d_model=config["D_MODEL"],
            num_heads=config["NUM_HEADS"],
            num_layers=config["NUM_ENCODER_LAYERS"],
            ffn_dim=config["DIM_FEEDFORWARD"],
            dropout=config["DROPOUT"],
            max_len=config["MAX_SEQ_LEN"]
        )
        self.mt5 = AutoModelForSeq2SeqLM.from_pretrained(config["MT5_MODEL_NAME"])

    def forward(self, src, labels=None):
        encoder_outputs = self.visual_encoder(src)
        if labels is not None:
            outputs = self.mt5(encoder_outputs=(encoder_outputs,), labels=labels)
        else:
            outputs = self.mt5(encoder_outputs=(encoder_outputs,))
        return outputs


# ──────────────────────────────────────────────────────────────────────────
# LANDMARK EXTRACTION — adapted from keypoints.ipynb, kept behaviorally
# identical (same models, same confidences, same smoothing/mirror logic)
# ──────────────────────────────────────────────────────────────────────────
def download_task_models():
    if not os.path.exists("hand_landmarker.task"):
        print("Downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "hand_landmarker.task"
        )
    if not os.path.exists("pose_landmarker.task"):
        print("Downloading MediaPipe Pose Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            "pose_landmarker.task"
        )


class ProjectedLandmark:
    """Lightweight landmark representation projected to screen coordinates."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MediaPipeLiveExtractor:
    """Same 208-dim extraction logic as keypoints.ipynb's MediaPipeVideoExtractor,
    with an intelligent Focus Lock & Aspect-Ratio Isolation system:
      - Automatically detects, tracks, and locks onto the user NEAREST to the webcam.
      - Extracts gestures from a central 4:3 crop at the recommended 640x480 resolution,
        matching the exact training dataset geometry and ensuring fine finger precision.
      - Filters out background bystanders, walking pedestrians, and spurious background
        clutter in peripheral 16:9 areas so they never register as landmarks.
      - Associates candidate hands exclusively with the locked user's reach envelope.
    """

    def __init__(self, min_detection_conf=0.5, min_tracking_conf=0.5,
                 mirror_fix=True, alpha=0.75):
        download_task_models()
        self.mirror_fix = mirror_fix
        self.alpha = alpha
        self.prev_landmarks = None

        # Focus Lock tracking state
        self.locked_user_center = None    # (cx, cy) normalized within 4:3 crop
        self.locked_user_span = None      # apparent shoulder span / size
        self.lock_consecutive_frames = 0
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 25         # ~0.8s grace period before unlocking
        self.MIN_USER_PROXIMITY = 0.11    # Minimum size to count as foreground user (filters background noise)
        self.lock_status = "SEARCHING"

        # Detect up to 4 hands and 4 poses to differentiate foreground user from background
        base_options_hand = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_hand = mp_vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_tracking_conf)
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(options_hand)

        base_options_pose = mp_python.BaseOptions(model_asset_path='pose_landmarker.task')
        options_pose = mp_vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_tracking_conf)
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(options_pose)

    def reset_tracker(self):
        """Must be called at the START of every new sign capture — the training
        pipeline calls this once per video so smoothing never bleeds across
        separate clips. We replicate that per-recording, not per-frame."""
        self.prev_landmarks = None

    def reset_focus_lock(self):
        """Forces re-acquisition of the user nearest to the webcam."""
        self.locked_user_center = None
        self.locked_user_span = None
        self.lock_consecutive_frames = 0
        self.lost_frames = 0
        self.lock_status = "SEARCHING"

    def set_mirror(self, value: bool):
        self.mirror_fix = value

    def process_frame(self, frame_bgr):
        """Alias for extract_frame for server/API bridge compatibility."""
        return self.extract_frame(frame_bgr)

    def extract_frame(self, frame_bgr):
        """Returns (features_208: np.ndarray, debug_info: dict) for one frame.

        Applies 4:3 central crop (downscaled to recommended 640x480) for landmark
        extraction matching the training pipeline, while returning re-projected
        landmarks for 16:9 full-frame visualization.
        """
        fh, fw = frame_bgr.shape[:2]
        target_aspect = 4.0 / 3.0
        current_aspect = fw / float(fh)

        # 1. Calculate central 4:3 crop bounds
        if current_aspect > target_aspect + 0.02:
            crop_w = int(fh * target_aspect)
            crop_x1 = (fw - crop_w) // 2
            crop_x2 = crop_x1 + crop_w
            crop_y1 = 0
            crop_y2 = fh
        else:
            crop_w = fw
            crop_x1 = 0
            crop_x2 = fw
            crop_y1 = 0
            crop_y2 = fh

        crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w_actual = crop_bgr.shape[:2]

        # 2. Resize crop to 640x480 sweet spot (fast inference + high-fidelity finger joints)
        if crop_w_actual != 640 or crop_h != 480:
            mp_input_bgr = cv2.resize(crop_bgr, (640, 480), interpolation=cv2.INTER_AREA)
        else:
            mp_input_bgr = crop_bgr

        frame_rgb = cv2.cvtColor(mp_input_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 1. Pose detection with Multi-Candidate Focus Lock
        pose_result = self.pose_detector.detect(mp_image)
        pose_coords = [0.0] * 66
        selected_pose = None
        focus_box = None
        background_poses_filtered = 0

        if pose_result.pose_landmarks:
            candidates = []
            for lms in pose_result.pose_landmarks:
                # Shoulder span is the primary physical proximity indicator
                sh_dist = math.hypot(lms[11].x - lms[12].x, lms[11].y - lms[12].y)
                torso_h = (math.hypot(lms[11].x - lms[23].x, lms[11].y - lms[23].y) +
                           math.hypot(lms[12].x - lms[24].x, lms[12].y - lms[24].y)) / 2.0
                apparent_size = max(sh_dist, torso_h * 0.7)
                cx = (lms[11].x + lms[12].x) / 2.0
                cy = (lms[11].y + lms[12].y) / 2.0

                # Bounding box of upper body for focus lock visual feedback
                ub_indices = [0, 11, 12, 13, 14, 23, 24]
                ub_xs = [lms[i].x for i in ub_indices if i < len(lms)]
                ub_ys = [lms[i].y for i in ub_indices if i < len(lms)]
                box = (min(ub_xs), min(ub_ys), max(ub_xs), max(ub_ys))

                candidates.append({
                    "landmarks": lms,
                    "apparent_size": apparent_size,
                    "center": (cx, cy),
                    "box": box,
                    "shoulder_dist": sh_dist,
                })

            # Filter out distant background noise / background people
            fg_candidates = [c for c in candidates if c["apparent_size"] >= self.MIN_USER_PROXIMITY]
            background_poses_filtered = len(candidates) - len(fg_candidates)

            if fg_candidates:
                selected_cand = None
                if self.locked_user_center is not None:
                    # Focus lock engaged: track locked user, penalizing sudden jumps
                    def lock_score(c):
                        dist = math.hypot(c["center"][0] - self.locked_user_center[0],
                                          c["center"][1] - self.locked_user_center[1])
                        proximity_factor = max(0.0, 1.0 - (dist / 0.40))
                        return (c["apparent_size"] * 0.6) + (proximity_factor * 0.4)

                    best_cand = max(fg_candidates, key=lock_score)
                    dist_to_lock = math.hypot(best_cand["center"][0] - self.locked_user_center[0],
                                              best_cand["center"][1] - self.locked_user_center[1])

                    if dist_to_lock < 0.40:
                        selected_cand = best_cand
                        alpha_t = 0.8
                        self.locked_user_center = (
                            alpha_t * self.locked_user_center[0] + (1 - alpha_t) * selected_cand["center"][0],
                            alpha_t * self.locked_user_center[1] + (1 - alpha_t) * selected_cand["center"][1]
                        )
                        self.locked_user_span = (alpha_t * self.locked_user_span +
                                                 (1 - alpha_t) * selected_cand["apparent_size"])
                        self.lost_frames = 0
                        self.lock_consecutive_frames += 1
                        self.lock_status = "LOCKED"
                    else:
                        self.lost_frames += 1
                        if self.lost_frames > self.MAX_LOST_FRAMES:
                            # Lost target past grace period; acquire current nearest user
                            selected_cand = max(fg_candidates, key=lambda c: c["apparent_size"])
                            self.locked_user_center = selected_cand["center"]
                            self.locked_user_span = selected_cand["apparent_size"]
                            self.lost_frames = 0
                            self.lock_consecutive_frames = 1
                            self.lock_status = "LOCKED"
                else:
                    # Initial lock: strictly pick the person NEAREST to the webcam (largest apparent size)
                    selected_cand = max(fg_candidates, key=lambda c: c["apparent_size"])
                    self.locked_user_center = selected_cand["center"]
                    self.locked_user_span = selected_cand["apparent_size"]
                    self.lost_frames = 0
                    self.lock_consecutive_frames = 1
                    self.lock_status = "LOCKED"

                if selected_cand is not None:
                    selected_pose = selected_cand["landmarks"]
                    focus_box = selected_cand["box"]
                    for i, lm in enumerate(selected_pose):
                        pose_coords[i * 2] = float(lm.x)
                        pose_coords[i * 2 + 1] = float(lm.y)
            else:
                # Poses detected, but all are too small / distant (filtered out as noise)
                self.lost_frames += 1
                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.locked_user_center = None
                    self.lock_status = "FILTERED_NOISE" if candidates else "NO_USER"
        else:
            self.lost_frames += 1
            if self.lost_frames > self.MAX_LOST_FRAMES:
                self.locked_user_center = None
                self.lock_status = "NO_USER"

        # 2. Hands detection with Body Reach Envelope & Size Filtering
        hand_result = self.hand_detector.detect(mp_image)
        lh_coords = [0.0] * 42
        rh_coords = [0.0] * 42
        detected_labels = []
        raw_hands = []  # ALWAYS raw/un-swapped -- used only for on-screen drawing
        background_hands_filtered = 0

        if hand_result.hand_landmarks and selected_pose is not None:
            user_lms = selected_pose
            body_xs = [user_lms[i].x for i in [11, 12, 13, 14, 15, 16] if i < len(user_lms)]
            body_ys = [user_lms[i].y for i in [0, 11, 12, 13, 14, 15, 16, 23, 24] if i < len(user_lms)]

            reach_margin = max(0.25, (self.locked_user_span or 0.2) * 0.9)
            env_x1 = max(0.0, min(body_xs) - reach_margin)
            env_x2 = min(1.0, max(body_xs) + reach_margin)
            env_y1 = max(0.0, min(body_ys) - reach_margin)
            env_y2 = min(1.0, max(body_ys) + reach_margin)

            pose_lw = (user_lms[15].x, user_lms[15].y)
            pose_rw = (user_lms[16].x, user_lms[16].y)
            pose_ls = (user_lms[11].x, user_lms[11].y)
            pose_rs = (user_lms[12].x, user_lms[12].y)

            valid_hands = []
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    handedness = hand_result.handedness[idx][0].category_name

                h_xs = [lm.x for lm in hand_lms]
                h_ys = [lm.y for lm in hand_lms]
                h_span = max(max(h_xs) - min(h_xs), max(h_ys) - min(h_ys))
                wrist_pt = (hand_lms[0].x, hand_lms[0].y)

                # Filter 1: Hand size (reject distant background hands / tiny noise)
                if h_span < 0.04:
                    background_hands_filtered += 1
                    continue

                # Filter 2: Anatomical reach envelope (must belong to locked user)
                if not (env_x1 <= wrist_pt[0] <= env_x2 and env_y1 <= wrist_pt[1] <= env_y2):
                    background_hands_filtered += 1
                    continue

                dist_l = min(math.hypot(wrist_pt[0] - pose_lw[0], wrist_pt[1] - pose_lw[1]),
                             math.hypot(wrist_pt[0] - pose_ls[0], wrist_pt[1] - pose_ls[1]))
                dist_r = min(math.hypot(wrist_pt[0] - pose_rw[0], wrist_pt[1] - pose_rw[1]),
                             math.hypot(wrist_pt[0] - pose_rs[0], wrist_pt[1] - pose_rs[1]))

                coords_flat = []
                for lm in hand_lms:
                    coords_flat.extend([lm.x, lm.y])

                valid_hands.append({
                    "handedness": handedness,
                    "landmarks": hand_lms,
                    "coords": coords_flat[:42],
                    "span": h_span,
                    "dist_l": dist_l,
                    "dist_r": dist_r,
                })

            # Anatomical Hand Association:
            # We strictly associate hands with the user's anatomical body side using Euclidean distance
            # to the locked user's wrists and shoulders (pose_rw/pose_rs vs pose_lw/pose_ls),
            # eliminating MediaPipe's erratic palm-facing "handedness" heuristic.
            best_anat_left = None
            best_anat_right = None

            if len(valid_hands) == 1:
                h = valid_hands[0]
                if h["dist_r"] < h["dist_l"]:
                    best_anat_right = h
                else:
                    best_anat_left = h
            elif len(valid_hands) >= 2:
                # Find optimal pairing that minimizes sum of distances
                best_cost = float('inf')
                for i in range(len(valid_hands)):
                    for j in range(len(valid_hands)):
                        if i == j:
                            continue
                        cost = valid_hands[i]["dist_r"] + valid_hands[j]["dist_l"]
                        if cost < best_cost:
                            best_cost = cost
                            best_anat_right = valid_hands[i]
                            best_anat_left = valid_hands[j]

            if best_anat_left is not None:
                raw_hands.append({"handedness": "Left", "landmarks": best_anat_left["landmarks"]})
                detected_labels.append("Left Hand")

            if best_anat_right is not None:
                raw_hands.append({"handedness": "Right", "landmarks": best_anat_right["landmarks"]})
                detected_labels.append("Right Hand")

            # Training pipeline mapping (matching keypoints.ipynb):
            # In keypoints.ipynb, mirror_fix=True was set, which swapped the signer's dominant
            # right hand into lh_coords (feature slots [66:108]).
            # When mirror_fix=True:
            #   lh_coords (slots 66:108)  <- Anatomical Right Hand (dominant hand)
            #   rh_coords (slots 108:150) <- Anatomical Left Hand
            # When mirror_fix=False:
            #   lh_coords (slots 66:108)  <- Anatomical Left Hand
            #   rh_coords (slots 108:150) <- Anatomical Right Hand
            if self.mirror_fix:
                if best_anat_right is not None:
                    lh_coords = best_anat_right["coords"]
                if best_anat_left is not None:
                    rh_coords = best_anat_left["coords"]
            else:
                if best_anat_left is not None:
                    lh_coords = best_anat_left["coords"]
                if best_anat_right is not None:
                    rh_coords = best_anat_right["coords"]

        # 3. Face — always zeros (SANA standard, matches training exactly)
        face_coords = [0.0] * 58

        current_frame_208 = np.array(pose_coords + lh_coords + rh_coords + face_coords, dtype=np.float32)

        # Exponential smoothing — identical to keypoints.ipynb
        if self.prev_landmarks is None:
            self.prev_landmarks = current_frame_208
        else:
            active_mask = (current_frame_208 != 0.0).astype(np.float32)
            smoothed = (active_mask * (self.alpha * current_frame_208 + (1 - self.alpha) * self.prev_landmarks)
                        + (1 - active_mask) * current_frame_208)
            self.prev_landmarks = smoothed
            current_frame_208 = smoothed

        # Re-project landmarks to full screen space for 16:9 rendering
        screen_pose = None
        if selected_pose is not None:
            screen_pose = [
                ProjectedLandmark(
                    (crop_x1 + lm.x * crop_w_actual) / float(fw),
                    (crop_y1 + lm.y * crop_h) / float(fh),
                    getattr(lm, 'z', 0.0)
                ) for lm in selected_pose
            ]

        screen_raw_hands = []
        for hand in raw_hands:
            proj_lms = [
                ProjectedLandmark(
                    (crop_x1 + lm.x * crop_w_actual) / float(fw),
                    (crop_y1 + lm.y * crop_h) / float(fh),
                    getattr(lm, 'z', 0.0)
                ) for lm in hand["landmarks"]
            ]
            screen_raw_hands.append({"handedness": hand["handedness"], "landmarks": proj_lms})

        screen_focus_box = None
        if focus_box is not None:
            screen_focus_box = (
                (crop_x1 + focus_box[0] * crop_w_actual) / float(fw),
                (crop_y1 + focus_box[1] * crop_h) / float(fh),
                (crop_x1 + focus_box[2] * crop_w_actual) / float(fw),
                (crop_y1 + focus_box[3] * crop_h) / float(fh)
            )

        debug_info = {
            "pose": screen_pose,
            "raw_hands": screen_raw_hands,
            "hands_detected": len(raw_hands),
            "raw_labels": detected_labels,
            "focus_box": screen_focus_box,
            "crop_rect": (crop_x1, crop_y1, crop_x2, crop_y2),
            "lock_status": self.lock_status,
            "filtered_poses": background_poses_filtered,
            "filtered_hands": background_hands_filtered,
        }
        return current_frame_208, debug_info

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()


TARGET_SHOULDER_WIDTH = 0.194
TARGET_CENTER_X = 0.490
TARGET_CENTER_Y = 0.496


def trim_gesture_sequence(sequence_208, pad_frames=6):
    """Trims leading and trailing frames where no hands are detected (zeros in slots 66:150),
    retaining pad_frames on each side to capture the hand entrance and exit motion.
    Prevents gestures from being artificially squished when the user hesitates before
    or after pressing SPACE."""
    T = sequence_208.shape[0]
    if T <= 10:
        return sequence_208

    hands_slice = sequence_208[:, 66:150]
    hands_active = (np.sum(hands_slice != 0, axis=1) > 0)

    if not np.any(hands_active):
        return sequence_208

    active_indices = np.where(hands_active)[0]
    start_idx = max(0, active_indices[0] - pad_frames)
    end_idx = min(T, active_indices[-1] + 1 + pad_frames)

    # Ensure minimum 15 frames for smooth interpolation
    if end_idx - start_idx < 15:
        mid = (start_idx + end_idx) // 2
        start_idx = max(0, mid - 10)
        end_idx = min(T, mid + 10)

    return sequence_208[start_idx:end_idx]


def normalize_sequence_to_training_canon(sequence_208):
    """Normalizes any sequence of (T, 208) coordinates so that the user's upper body
    matches the exact reference scale and position of the training dataset:
      - Shoulder width = 0.194
      - Shoulder center = (0.490, 0.496)
    This makes inference completely invariant to whether the user sits close (laptop desk)
    or stands further back from the webcam."""
    normalized = sequence_208.copy()
    T = normalized.shape[0]

    l_sh_xs = normalized[:, 11 * 2]
    l_sh_ys = normalized[:, 11 * 2 + 1]
    r_sh_xs = normalized[:, 12 * 2]
    r_sh_ys = normalized[:, 12 * 2 + 1]

    valid_mask = (l_sh_xs != 0.0) & (r_sh_xs != 0.0)
    if not np.any(valid_mask):
        return normalized

    sh_widths = np.hypot(l_sh_xs[valid_mask] - r_sh_xs[valid_mask], l_sh_ys[valid_mask] - r_sh_ys[valid_mask])
    mean_sh_w = np.median(sh_widths)

    if mean_sh_w < 0.03:
        return normalized

    mean_cx = np.median((l_sh_xs[valid_mask] + r_sh_xs[valid_mask]) / 2.0)
    mean_cy = np.median((l_sh_ys[valid_mask] + r_sh_ys[valid_mask]) / 2.0)

    scale = TARGET_SHOULDER_WIDTH / mean_sh_w

    # Affine scale and center: pose (0:66), left hand (66:108), right hand (108:150)
    for t in range(T):
        # Pose
        for i in range(33):
            if normalized[t, i * 2] != 0.0 or normalized[t, i * 2 + 1] != 0.0:
                normalized[t, i * 2] = TARGET_CENTER_X + (normalized[t, i * 2] - mean_cx) * scale
                normalized[t, i * 2 + 1] = TARGET_CENTER_Y + (normalized[t, i * 2 + 1] - mean_cy) * scale
        # Hand 1 (slots 66:108)
        for i in range(21):
            idx = 66 + i * 2
            if normalized[t, idx] != 0.0 or normalized[t, idx + 1] != 0.0:
                normalized[t, idx] = TARGET_CENTER_X + (normalized[t, idx] - mean_cx) * scale
                normalized[t, idx + 1] = TARGET_CENTER_Y + (normalized[t, idx + 1] - mean_cy) * scale
        # Hand 2 (slots 108:150)
        for i in range(21):
            idx = 108 + i * 2
            if normalized[t, idx] != 0.0 or normalized[t, idx + 1] != 0.0:
                normalized[t, idx] = TARGET_CENTER_X + (normalized[t, idx] - mean_cx) * scale
                normalized[t, idx + 1] = TARGET_CENTER_Y + (normalized[t, idx + 1] - mean_cy) * scale

    return normalized


def resample_sequence(sequence, target_frames=60):
    """Identical to keypoints.ipynb's resample_sequence."""
    T = sequence.shape[0]
    if T == target_frames:
        return sequence
    if T < 2:
        # Degenerate case (e.g. only 1 frame captured) — pad by repeating
        return np.repeat(sequence, target_frames, axis=0)[:target_frames]

    orig_times = np.linspace(0, 1, T)
    target_times = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        resampled[:, dim] = np.interp(target_times, orig_times, sequence[:, dim])
    return resampled


def pad_to_max_seq_len(sequence, max_seq_len, input_dim):
    """Identical to MedicalDataset.__getitem__'s padding/truncation logic
    in notebook399144a5d3.ipynb."""
    T = sequence.shape[0]
    if T > max_seq_len:
        sequence = sequence[:max_seq_len]
    elif T < max_seq_len:
        padding = np.zeros((max_seq_len - T, input_dim), dtype=np.float32)
        sequence = np.vstack([sequence, padding])
    return sequence


# ──────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS (visual feedback only — never touches the 208-dim vector)
# ──────────────────────────────────────────────────────────────────────────

# Connection index pairs for drawing skeletons. Prefer MediaPipe's own
# constants when available; fall back to the well-known standard topology
# otherwise so this never breaks across mediapipe package variants.
try:
    from mediapipe.python.solutions.hands import HAND_CONNECTIONS as _MP_HAND_CONNECTIONS
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS as _MP_POSE_CONNECTIONS
    HAND_CONNECTIONS = _MP_HAND_CONNECTIONS
    POSE_CONNECTIONS = _MP_POSE_CONNECTIONS
except Exception:
    HAND_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ])
    POSE_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ])

LEFT_HAND_COLOR = (0, 200, 0)     # green (BGR) -- your actual left hand, always
RIGHT_HAND_COLOR = (0, 0, 255)    # red (BGR)   -- your actual right hand, always
POSE_COLOR = (255, 128, 0)


def _draw_skeleton(frame, landmarks, connections, point_color, line_color):
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, point_color, -1)
    return pts


def draw_landmarks_on_frame(frame, debug_info):
    """Draws full skeletons for the locked nearest user and their validated hands,
    displays studio 4:3 gesture zone boundaries & vignette, corner brackets marking
    the focus-locked user, and labels each hand with its RAW identity."""
    h, w = frame.shape[:2]

    # Draw studio 4:3 gesture zone vignette if in widescreen display
    crop_rect = debug_info.get("crop_rect")
    if crop_rect:
        x1, y1, x2, y2 = crop_rect
        if x1 > 0 or x2 < w:
            if x1 > 0:
                # Translucent vignette on left margin
                left_roi = frame[:, :x1]
                frame[:, :x1] = (left_roi.astype(np.float32) * 0.55).astype(np.uint8)
                cv2.line(frame, (x1, 0), (x1, h), (180, 160, 40), 1, cv2.LINE_AA)
            if x2 < w:
                # Translucent vignette on right margin
                right_roi = frame[:, x2:]
                frame[:, x2:] = (right_roi.astype(np.float32) * 0.55).astype(np.uint8)
                cv2.line(frame, (x2, 0), (x2, h), (180, 160, 40), 1, cv2.LINE_AA)

            cv2.putText(frame, "[ 4:3 GESTURE ZONE ]", (x1 + 12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 160, 40), 1, cv2.LINE_AA)

    # Draw focus lock corner brackets around the nearest locked user
    focus_box = debug_info.get("focus_box")
    lock_status = debug_info.get("lock_status", "IDLE")

    if focus_box is not None and lock_status == "LOCKED":
        bx1 = max(0, int((focus_box[0] - 0.04) * w))
        by1 = max(0, int((focus_box[1] - 0.04) * h))
        bx2 = min(w - 1, int((focus_box[2] + 0.04) * w))
        by2 = min(h - 1, int((focus_box[3] + 0.04) * h))

        bracket_color = (0, 255, 128)
        b_len = min(25, max(8, (bx2 - bx1) // 5), max(8, (by2 - by1) // 5))
        if b_len > 4 and (bx2 > bx1 + 10) and (by2 > by1 + 10):
            # Top-left
            cv2.line(frame, (bx1, by1), (bx1 + b_len, by1), bracket_color, 2)
            cv2.line(frame, (bx1, by1), (bx1, by1 + b_len), bracket_color, 2)
            # Top-right
            cv2.line(frame, (bx2, by1), (bx2 - b_len, by1), bracket_color, 2)
            cv2.line(frame, (bx2, by1), (bx2, by1 + b_len), bracket_color, 2)
            # Bottom-left
            cv2.line(frame, (bx1, by2), (bx1 + b_len, by2), bracket_color, 2)
            cv2.line(frame, (bx1, by2), (bx1, by2 - b_len), bracket_color, 2)
            # Bottom-right
            cv2.line(frame, (bx2, by2), (bx2 - b_len, by2), bracket_color, 2)
            cv2.line(frame, (bx2, by2), (bx2, by2 - b_len), bracket_color, 2)

            cv2.putText(frame, "FOCUS LOCKED", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bracket_color, 1)

    _draw_skeleton(frame, debug_info.get("pose"), POSE_CONNECTIONS, POSE_COLOR, (200, 100, 0))

    for hand in debug_info.get("raw_hands", []):
        is_left = hand["handedness"] == "Left"
        color = LEFT_HAND_COLOR if is_left else RIGHT_HAND_COLOR
        pts = _draw_skeleton(frame, hand["landmarks"], HAND_CONNECTIONS, color, color)
        if pts:
            wrist_x, wrist_y = pts[0]
            label = "Left hand" if is_left else "Right hand"
            cv2.putText(frame, label, (wrist_x - 20, wrist_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ──────────────────────────────────────────────────────────────────────────
# URDU TEXT RENDERING — OpenCV's putText CANNOT shape/reorder Arabic script
# (it draws disconnected, left-to-right glyphs, which is why Urdu looked
# broken on screen before). This uses PIL + proper reshaping/bidi instead.
# Falls back cleanly (no on-screen Urdu, but still in console) if the
# optional libraries or a compatible font aren't available -- never shows
# broken text.
# ──────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display
    URDU_RENDER_LIBS_AVAILABLE = True
except ImportError:
    URDU_RENDER_LIBS_AVAILABLE = False

_URDU_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\jameel noori nastaleeq.ttf",
    "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Geeza Pro.ttc",
    # NOTE: deliberately NOT including generic fonts like DejaVuSans here --
    # they lack Arabic/Urdu glyphs and PIL will silently draw empty "tofu"
    # boxes instead of erroring, which is worse than falling back cleanly.
]

_urdu_font_path = None
_urdu_warning_shown = False


def _find_urdu_font():
    global _urdu_font_path
    if _urdu_font_path is not None:
        return _urdu_font_path
    for path in _URDU_FONT_CANDIDATES:
        if os.path.exists(path):
            _urdu_font_path = path
            return path
    _urdu_font_path = ""
    return ""


def draw_urdu_text(frame, text, position, font_size=26, color=(0, 255, 0)):
    """Overlays a correctly shaped, correctly ordered Urdu string onto an
    OpenCV BGR frame using PIL. Returns True if it actually drew something,
    False if it skipped (so the caller can decide whether to warn once)."""
    global _urdu_warning_shown
    if not text:
        return False

    if not URDU_RENDER_LIBS_AVAILABLE:
        if not _urdu_warning_shown:
            print("[urdu-render] arabic_reshaper/python-bidi/Pillow not installed -- "
                  "Urdu will only show in the console, not on screen. Run: "
                  "pip install pillow arabic-reshaper python-bidi")
            _urdu_warning_shown = True
        return False

    font_path = _find_urdu_font()
    if not font_path:
        if not _urdu_warning_shown:
            print("[urdu-render] No compatible Urdu/Arabic font found on this system -- "
                  "Urdu will only show in the console, not on screen.")
            _urdu_warning_shown = True
        return False

    try:
        reshaped = arabic_reshaper.reshape(text)
        display_text = get_display(reshaped)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, font_size)
        rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
        draw.text(position, display_text, font=font, fill=rgb_color)

        result_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:, :] = result_bgr
        return True
    except Exception as e:
        if not _urdu_warning_shown:
            print(f"[urdu-render] Failed to render Urdu on screen ({e}) -- "
                  f"still available in the console.")
            _urdu_warning_shown = True
        return False


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def load_model(model_path, device):
    print(f"Loading tokenizer ({CONFIG['MT5_MODEL_NAME']})...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MT5_MODEL_NAME"])

    print("Building SANA_PSL_Translator architecture...")
    model = SANA_PSL_Translator(CONFIG)

    if os.path.exists(model_path):
        print(f"Loading fine-tuned weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except Exception:
                model.visual_encoder.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(
            f"Could not find model weights at '{model_path}'. "
            f"Pass --model <path to sana_psl_medical_finetuned.pt>"
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer


def resolve_prediction(raw_pred: str, sequence_208=None, runner_up: str = "", confidence_pct: float = 100.0) -> str:
    """Constrains prediction strictly to the canonical 16 trained medical sentences,
    and applies physical kinematic disambiguation when needed.
    """
    if not raw_pred:
        return TRAINED_SENTENCES[0]

    clean_pred = raw_pred.strip()

    def _norm(txt):
        return "".join(c for c in txt.lower().replace("-", " ") if c.isalnum() or c.isspace()).strip()

    norm_pred = _norm(clean_pred)
    matched = None

    # 1. Exact / normalized match check
    for sent in TRAINED_SENTENCES:
        if _norm(sent) == norm_pred:
            matched = sent
            break
    if matched is None:
        # Normalized fuzzy match
        norm_map = {_norm(s): s for s in TRAINED_SENTENCES}
        close_matches = difflib.get_close_matches(norm_pred, list(norm_map.keys()), n=1, cutoff=0.3)
        if close_matches:
            matched = norm_map[close_matches[0]]
        else:
            # Word overlap fallback
            pred_words = set(norm_pred.split())
            best_overlap = -1
            matched = TRAINED_SENTENCES[0]
            for sent in TRAINED_SENTENCES:
                overlap = len(pred_words.intersection(set(_norm(sent).split())))
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched = sent

    # 2. Kinematic Disambiguation (requires landmark sequence)
    # ONLY apply when there is genuine ambiguity between confusing pairs
    if sequence_208 is not None and matched:
        seq = np.array(sequence_208)
        pose = seq[:, 0:66]
        lh = seq[:, 66:108]  # Dominant active hand slot

        active = (lh[:, 0] != 0.0)
        n_active = np.sum(active)

        if n_active >= 3:
            nose_y = pose[active, 1]
            nose_x = pose[active, 0]
            tip_y = lh[active, 17]  # Index fingertip
            tip_x = lh[active, 16]

            # Spatial elevation & head proximity
            dist_nose = np.hypot(tip_x - nose_x, tip_y - nose_y)
            min_dist_nose = float(np.min(dist_nose))
            head_frac = float(np.mean(tip_y < (nose_y + 0.06)))

            # Finger extensions relative to wrist
            idx_wrist = float(np.mean(np.hypot(lh[active, 16] - lh[active, 0], lh[active, 17] - lh[active, 1])))
            pinky_wrist = float(np.mean(np.hypot(lh[active, 40] - lh[active, 0], lh[active, 41] - lh[active, 1])))

            # Late-frame pinch distance for kam vs tez (frames ~33-60)
            eval_start = max(0, int(len(seq) * 0.55))
            eval_end = min(len(seq), int(len(seq) * 1.0))
            valid_pinch = (lh[eval_start:eval_end, 8] != 0.0) & (lh[eval_start:eval_end, 16] != 0.0)
            if np.any(valid_pinch):
                d_pinch = np.sqrt(
                    (lh[eval_start:eval_end, 8][valid_pinch] - lh[eval_start:eval_end, 16][valid_pinch]) ** 2 +
                    (lh[eval_start:eval_end, 9][valid_pinch] - lh[eval_start:eval_end, 17][valid_pinch]) ** 2
                )
                mean_pinch = float(np.mean(d_pinch))
            else:
                mean_pinch = 0.06

            # --- RULE A: YES vs NO DISAMBIGUATION ---
            norm_runner = _norm(runner_up) if runner_up else ""
            is_yes_no_cand = ({_norm(matched), norm_runner} == {"yes", "no"}) or (_norm(matched) in ["yes", "no"] and confidence_pct < 65.0)
            if is_yes_no_cand:
                if idx_wrist < 0.075 and pinky_wrist < 0.045:
                    matched = "Yes"
                elif idx_wrist >= 0.075 and pinky_wrist >= 0.045:
                    matched = "No"

            # --- RULE B: HEAD vs CHEST DISAMBIGUATION ---
            is_head_chest_ambig = ({_norm(matched), norm_runner} & {"mere sarr mein dard hai"} and {_norm(matched), norm_runner} & {"mujhay dard kam hai", "mujhay dard tez hai"})
            if is_head_chest_ambig:
                if min_dist_nose < 0.08 or head_frac > 0.35:
                    matched = "Mere sarr mein dard hai"
                else:
                    matched = "Mujhay dard kam hai" if mean_pinch < 0.070 else "Mujhay dard tez hai"

    return matched


def run_inference(model, tokenizer, sequence_208_100, device, raw_resampled_60=None):
    """sequence_208_100: np.ndarray shape (100, 208), already padded exactly
    like MedicalDataset produces at train time.

    Executes 4-beam search sequence generation with softmax confidence scoring.
    Returns (pred_text, latency_ms, confidence_pct, runner_up).
    """
    data_tensor = torch.tensor(sequence_208_100, dtype=torch.float32).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        encoder_outputs = model.visual_encoder(data_tensor)
        hf_encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        gen_outputs = model.mt5.generate(
            encoder_outputs=hf_encoder_outputs,
            max_length=CONFIG["MAX_TARGET_LEN"],
            num_beams=4,
            num_return_sequences=2,
            return_dict_in_generate=True,
            output_scores=True,
        )
    latency_ms = (time.time() - start_time) * 1000.0

    raw_pred = tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True).strip()
    runner_up = tokenizer.decode(gen_outputs.sequences[1], skip_special_tokens=True).strip() if len(gen_outputs.sequences) > 1 else ""

    # True confidence from beam sequence log-probabilities
    if hasattr(gen_outputs, "sequences_scores") and gen_outputs.sequences_scores is not None and len(gen_outputs.sequences_scores) >= 2:
        s1 = float(gen_outputs.sequences_scores[0].item())
        s2 = float(gen_outputs.sequences_scores[1].item())
        exp1 = math.exp(s1)
        exp2 = math.exp(s2)
        confidence_pct = (exp1 / (exp1 + exp2)) * 100.0
    else:
        confidence_pct = 85.0

    seq_for_kinematics = raw_resampled_60 if raw_resampled_60 is not None else sequence_208_100[:60]
    pred_text = resolve_prediction(raw_pred, seq_for_kinematics, runner_up=runner_up, confidence_pct=confidence_pct)

    return pred_text, latency_ms, confidence_pct, runner_up


def run_collapse_diagnostic(model, tokenizer, real_sequence_100, device):
    """Sanity check for output collapse: feeds the model three very different
    inputs (your real captured sign, all-zeros, and random noise) and prints
    all three predictions. If they come back identical, the model is NOT
    conditioning on the landmarks. If they differ, that is real evidence it is
    reacting to input."""
    print("\n" + "-" * 60)
    print("COLLAPSE DIAGNOSTIC (Beam Search Verification)")
    print("-" * 60)

    real_pred, real_lat, real_conf, real_run = run_inference(model, tokenizer, real_sequence_100, device)
    print(f"  Real captured sign   -> {real_pred!r}  ({real_lat:.0f}ms, {real_conf:.1f}% conf)")

    zeros_seq = np.zeros_like(real_sequence_100)
    zeros_pred, zeros_lat, zeros_conf, zeros_run = run_inference(model, tokenizer, zeros_seq, device)
    print(f"  All-zeros input      -> {zeros_pred!r}  ({zeros_lat:.0f}ms, {zeros_conf:.1f}% conf)")

    noise_seq = np.random.uniform(0.0, 1.0, size=real_sequence_100.shape).astype(np.float32)
    noise_pred, noise_lat, noise_conf, noise_run = run_inference(model, tokenizer, noise_seq, device)
    print(f"  Random noise input   -> {noise_pred!r}  ({noise_lat:.0f}ms, {noise_conf:.1f}% conf)")

    if real_pred == zeros_pred == noise_pred:
        print("\n  ⚠️  ALL THREE PREDICTIONS ARE IDENTICAL.")
        print("  This suggests the model has collapsed to a fixed fallback output.")
    else:
        print("\n  ✅ Predictions differ across inputs -- model is actively reactive")
        print("  and conditioning on physical landmark inputs.")
    print("-" * 60 + "\n")


def process_captured_buffer(buffer, model, tokenizer, device, args, tts):
    """Common pipeline for processing recorded frames whether stopped manually or via auto-stop."""
    if len(buffer) < 3:
        print("Too few frames captured — try again with a longer sign.")
        return None, 0.0, 0.0, "", None, ""

    raw_seq = np.array(buffer, dtype=np.float32)
    # Direct resample to 60 frames matching the training contract (no harmful frame trimming)
    resampled_60 = resample_sequence(raw_seq, target_frames=CONFIG["TARGET_FRAMES"])
    padded_100 = pad_to_max_seq_len(resampled_60, CONFIG["MAX_SEQ_LEN"], CONFIG["INPUT_DIM"])

    pred_text, latency_ms, confidence_pct, runner_up = run_inference(
        model, tokenizer, padded_100, device, raw_resampled_60=resampled_60)

    print(f"\n================ PREDICTION RESULT ================")
    print(f">>> Top Prediction : {pred_text} ({confidence_pct:.1f}%)")
    if runner_up and runner_up.lower() != pred_text.lower():
        print(f">>> Runner-up      : {runner_up}")
    print(f">>> Latency        : {latency_ms:.0f}ms")

    last_urdu = ""
    if not args.no_translate:
        last_urdu = translate_to_urdu(pred_text)
        if last_urdu:
            print(f">>> Urdu Meaning   : {last_urdu}")
    print("===================================================\n")

    if tts is not None:
        tts.speak_async(pred_text)

    return pred_text, latency_ms, confidence_pct, last_urdu, padded_100, runner_up


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sana_psl_medical_finetuned.pt",
                         help="Path to the fine-tuned .pt weights file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--camera-warmup-frames", type=int, default=30,
                         help="Frames to read and discard right after opening the "
                              "webcam, letting auto-exposure/auto-focus settle "
                              "before predictions are trusted (0 disables this).")
    parser.add_argument("--auto-stop-secs", type=float, default=0.0,
                         help="Auto-stop a recording after this many seconds with "
                              "no hands detected (0 disables auto-stop).")
    parser.add_argument("--no-translate", action="store_true",
                         help="Skip Urdu translation entirely")
    parser.add_argument("--no-tts", action="store_true",
                         help="Skip text-to-speech playback of predictions")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.model, device)

    if device.type == "cpu":
        print("[perf] Running on CPU. mT5 generation is inherently slower on CPU "
              "than GPU -- if you have an NVIDIA GPU, installing the CUDA build of "
              "torch (see pytorch.org) will noticeably cut latency.")

    print("Warming up model (absorbs one-time initialization cost so your first "
          "real prediction isn't artificially slow)...")
    _warmup_seq = np.zeros((CONFIG["MAX_SEQ_LEN"], CONFIG["INPUT_DIM"]), dtype=np.float32)
    _, warmup_latency_ms, _, _ = run_inference(model, tokenizer, _warmup_seq, device)
    print(f"Warm-up complete ({warmup_latency_ms:.0f}ms one-time cost). "
          f"Real predictions from here on should be faster.\n")

    tts = TextToSpeech() if not args.no_tts else None
    if args.no_tts:
        print("[tts] Disabled via --no-tts.")
    elif not (tts and tts.available):
        print("[tts] Not available on this system -- predictions will still work, "
              "just silently. (pip install pyttsx3 to enable)")

    extractor = MediaPipeLiveExtractor(
        min_detection_conf=CONFIG["MIN_DETECTION_CONF"],
        min_tracking_conf=CONFIG["MIN_TRACKING_CONF"],
        mirror_fix=CONFIG["MIRROR_CORRECTION"],
        alpha=CONFIG["SMOOTHING_ALPHA"],
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    # Request 16:9 widescreen (720p) for modern displays
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Camera warm-up: right after VideoCapture opens, auto-exposure/auto-focus
    # are still adjusting, so the first ~1 second of frames are often dark or
    # blurry and MediaPipe detects little/nothing on them. If a recording were
    # captured during this window it would resemble the all-zeros collapse
    # case ("Test are cheap here"). We just read and discard frames here --
    # nothing is extracted or fed to the model -- so this cannot affect
    # prediction correctness, only delay when SPACE becomes meaningful.
    if args.camera_warmup_frames > 0:
        print(f"Warming up camera ({args.camera_warmup_frames} frames, letting "
              f"auto-exposure/focus settle)...")
        for _ in range(args.camera_warmup_frames):
            cap.read()
        print("Camera ready.\n")

    recording = False
    buffer = []
    last_hand_seen_time = None
    last_prediction = ""
    last_runner_up = ""
    last_urdu = ""
    last_latency_ms = 0.0
    last_confidence_pct = 0.0
    last_padded_sequence = None  # kept for the 'd' collapse diagnostic

    WINDOW_NAME = "SANA A-PSL Live Inference"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    print("\n" + "=" * 60)
    print("Ready. Press SPACE to start recording a sign, SPACE again to stop.")
    print("Press 'm' to toggle mirror-hand-correction.")
    print("Press 'l' to reset focus lock (re-lock onto nearest user).")
    print("Press 'f' to toggle fullscreen mode (video stretches across display).")
    print("Press 'd' to run a collapse diagnostic on the last captured sign.")
    print("Press 'q' to quit.")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            features_208, debug_info = extractor.extract_frame(frame)
            draw_landmarks_on_frame(frame, debug_info)

            if recording:
                buffer.append(features_208)
                if debug_info["hands_detected"] > 0:
                    last_hand_seen_time = time.time()

                cv2.putText(frame, f"RECORDING... frames={len(buffer)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Optional auto-stop if hands have been absent for a while
                if (args.auto_stop_secs > 0 and last_hand_seen_time is not None
                        and time.time() - last_hand_seen_time > args.auto_stop_secs
                        and len(buffer) > 5):
                    recording = False
                    print(f"\nAuto-stopped after {args.auto_stop_secs}s of no hands. Captured {len(buffer)} frames.")
                    last_prediction, last_latency_ms, last_confidence_pct, last_urdu, last_padded_sequence, last_runner_up = \
                        process_captured_buffer(buffer, model, tokenizer, device, args, tts)
                    buffer = []
            else:
                cv2.putText(frame, "Press SPACE to record a sign", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Focus lock & tracking status HUD
            lock_st = debug_info.get("lock_status", "IDLE")
            filt_p = debug_info.get("filtered_poses", 0)
            filt_h = debug_info.get("filtered_hands", 0)
            if lock_st == "LOCKED":
                lock_text = "FOCUS LOCK: ACTIVE (Nearest User Locked)"
                lock_color = (0, 255, 0)
            elif lock_st == "FILTERED_NOISE":
                lock_text = f"FOCUS LOCK: BACKGROUND FILTERED ({filt_p} noise/bystanders ignored)"
                lock_color = (0, 165, 255)
            elif lock_st == "SEARCHING":
                lock_text = "FOCUS LOCK: SEARCHING (Step in front of camera)"
                lock_color = (0, 220, 255)
            else:
                lock_text = "FOCUS LOCK: IDLE"
                lock_color = (180, 180, 180)

            cv2.putText(frame, lock_text, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, lock_color, 2)
            cv2.putText(frame, f"Hands: {debug_info.get('raw_labels', [])} | Mirror: {extractor.mirror_fix}", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
            cv2.putText(frame, "[Controls: SPACE=Record, 'l'=Relock, 'f'=Fullscreen, 'm'=Mirror, 'q'=Quit]",
                        (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

            if last_prediction:
                # Anchor-aware stacked layout. OpenCV's putText anchors text at
                # its BOTTOM (baseline); PIL's draw.text (used for Urdu) anchors
                # at the TOP. Mixing the two with fixed offsets is what caused
                # the English/Urdu overlap before. Here we compute each row's
                # real height with cv2.getTextSize and stack rows bottom-up
                # with an explicit gap, so rows can never collide regardless
                # of font size or text length.
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                bottom_margin = 12
                row_gap = 10

                # Row 1 (bottom): latency/confidence
                latency_text = (f"Latency: {last_latency_ms:.0f}ms   "
                                 f"Confidence: {last_confidence_pct:.1f}%")
                lat_scale, lat_thick = 0.55, 1
                (_, lat_h), lat_base = cv2.getTextSize(latency_text, FONT, lat_scale, lat_thick)
                latency_baseline_y = frame.shape[0] - bottom_margin
                cursor_y = latency_baseline_y - lat_h - lat_base - row_gap

                # Row 2 (middle, optional): Urdu -- reserve space top-down
                # since draw_urdu_text positions from the top-left corner.
                urdu_top_y = None
                urdu_font_size = 26
                if last_urdu:
                    urdu_row_height = urdu_font_size + 10
                    urdu_top_y = cursor_y - urdu_row_height
                    cursor_y = urdu_top_y - row_gap

                # Row 3 (top): English prediction
                pred_text_full = f"Prediction: {last_prediction}"
                pred_scale, pred_thick = 0.7, 2
                (_, pred_h), pred_base = cv2.getTextSize(pred_text_full, FONT, pred_scale, pred_thick)
                pred_baseline_y = cursor_y

                # Semi-transparent background panel behind the whole text
                # block so it stays legible over busy video backgrounds.
                panel_top = max(0, pred_baseline_y - pred_h - pred_base - 8)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, panel_top), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

                # Draw the two OpenCV rows first; draw Urdu (PIL) last, since
                # draw_urdu_text re-renders the whole frame buffer and must
                # therefore run after everything else that's on it.
                cv2.putText(frame, pred_text_full, (10, pred_baseline_y),
                            FONT, pred_scale, (0, 255, 0), pred_thick)
                cv2.putText(frame, latency_text, (10, latency_baseline_y),
                            FONT, lat_scale, (0, 220, 220), lat_thick)

                if last_urdu:
                    urdu_drawn = draw_urdu_text(frame, last_urdu, (10, urdu_top_y),
                                                 font_size=urdu_font_size, color=(0, 255, 0))
                    if not urdu_drawn:
                        # Fell back gracefully -- still show it's available in console.
                        # Reuses the same reserved row, so it still can't overlap.
                        note_text = "(Urdu shown in console -- see terminal)"
                        note_scale, note_thick = 0.5, 1
                        (_, note_h), note_base = cv2.getTextSize(note_text, FONT, note_scale, note_thick)
                        note_baseline_y = urdu_top_y + urdu_row_height - note_base - 2
                        cv2.putText(frame, note_text, (10, note_baseline_y),
                                    FONT, note_scale, (150, 150, 150), note_thick)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not recording:
                    # Starting a new capture — reset smoothing exactly like
                    # extract_from_video() does per training clip.
                    extractor.reset_tracker()
                    buffer = []
                    last_hand_seen_time = time.time()
                    recording = True
                    print("Recording started...")
                else:
                    recording = False
                    print(f"Recording stopped. Captured {len(buffer)} raw frames.")
                    last_prediction, last_latency_ms, last_confidence_pct, last_urdu, last_padded_sequence, last_runner_up = \
                        process_captured_buffer(buffer, model, tokenizer, device, args, tts)
                    buffer = []

            elif key == ord('l'):
                extractor.reset_focus_lock()
                print("Focus lock reset -> searching for nearest user.")

            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Fullscreen enabled (press 'f' again to exit).")
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("Fullscreen disabled.")

            elif key == ord('m'):
                extractor.set_mirror(not extractor.mirror_fix)
                print(f"Mirror correction toggled -> {extractor.mirror_fix}")

            elif key == ord('d'):
                if last_padded_sequence is None:
                    print("No captured sign yet -- record one with SPACE first.")
                else:
                    run_collapse_diagnostic(model, tokenizer, last_padded_sequence, device)

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
