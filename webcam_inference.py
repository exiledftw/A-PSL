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

USAGE:
    python webcam_inference.py --model sana_psl_medical_finetuned.pt

CONTROLS (while the webcam window is focused):
    SPACE  - Start recording a sign. Press SPACE again to stop and run inference.
    m      - Toggle mirror-hand-correction (use if Left/Right hands look swapped
             on screen — see note below).
    q      - Quit.

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
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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

MEDICAL_DICTIONARY = {
    "Hello I need to see a doctor": "ہیلو، مجھے ڈاکٹر سے ملنے کی ضرورت ہے۔",
    "I have a severe headache": "مجھے شدید سر درد ہے۔",
    "Where is the pain": "درد کہاں ہے؟",
    "Are you having trouble breathing": "کیا آپ کو سانس لینے میں تکلیف ہو رہی ہے؟",
}


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


class MediaPipeLiveExtractor:
    """Same 208-dim extraction logic as keypoints.ipynb's MediaPipeVideoExtractor,
    plus it hands back the raw landmark objects so we can draw an overlay."""

    def __init__(self, min_detection_conf=0.5, min_tracking_conf=0.5,
                 mirror_fix=True, alpha=0.75):
        download_task_models()
        self.mirror_fix = mirror_fix
        self.alpha = alpha
        self.prev_landmarks = None

        base_options_hand = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_hand = mp_vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_tracking_conf)
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(options_hand)

        base_options_pose = mp_python.BaseOptions(model_asset_path='pose_landmarker.task')
        options_pose = mp_vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_tracking_conf)
        self.pose_detector = mp_vision.PoseLandmarker.create_from_options(options_pose)

    def reset_tracker(self):
        """Must be called at the START of every new sign capture — the training
        pipeline calls this once per video so smoothing never bleeds across
        separate clips. We replicate that per-recording, not per-frame."""
        self.prev_landmarks = None

    def set_mirror(self, value: bool):
        self.mirror_fix = value

    def extract_frame(self, frame_bgr):
        """Returns (features_208: np.ndarray, debug_info: dict) for one frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 1. Pose (33 pts -> 66 floats)
        pose_result = self.pose_detector.detect(mp_image)
        pose_coords = [0.0] * 66
        pose_landmarks_for_draw = None
        if pose_result.pose_landmarks:
            pose_landmarks_for_draw = pose_result.pose_landmarks[0]
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                pose_coords[i * 2] = float(lm.x)
                pose_coords[i * 2 + 1] = float(lm.y)

        # 2. Hands (21 pts/hand -> 42 floats/hand)
        hand_result = self.hand_detector.detect(mp_image)
        lh_coords = [0.0] * 42
        rh_coords = [0.0] * 42
        lh_landmarks_for_draw = None
        rh_landmarks_for_draw = None
        detected_labels = []

        if hand_result.hand_landmarks:
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    handedness = hand_result.handedness[idx][0].category_name
                detected_labels.append(handedness)

                coords_flat = []
                for lm in hand_lms:
                    coords_flat.extend([lm.x, lm.y])

                if handedness == "Left":
                    lh_coords = coords_flat[:42]
                    lh_landmarks_for_draw = hand_lms
                else:
                    rh_coords = coords_flat[:42]
                    rh_landmarks_for_draw = hand_lms

        # Mirror correction — identical logic to keypoints.ipynb
        if self.mirror_fix:
            lh_coords, rh_coords = rh_coords, lh_coords
            lh_landmarks_for_draw, rh_landmarks_for_draw = rh_landmarks_for_draw, lh_landmarks_for_draw

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

        debug_info = {
            "pose": pose_landmarks_for_draw,
            "left_hand": lh_landmarks_for_draw,
            "right_hand": rh_landmarks_for_draw,
            "hands_detected": len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0,
            "raw_labels": detected_labels,
        }
        return current_frame_208, debug_info

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()


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
def draw_landmarks_on_frame(frame, debug_info):
    h, w = frame.shape[:2]

    def draw_points(landmarks, color):
        if landmarks is None:
            return
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, color, -1)

    draw_points(debug_info.get("pose"), (255, 128, 0))       # blue-ish
    draw_points(debug_info.get("left_hand"), (0, 200, 0))    # green
    draw_points(debug_info.get("right_hand"), (0, 0, 255))   # red


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


def run_inference(model, tokenizer, sequence_208_100, device):
    """sequence_208_100: np.ndarray shape (100, 208), already padded exactly
    like MedicalDataset produces at train time."""
    data_tensor = torch.tensor(sequence_208_100, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs = model.visual_encoder(data_tensor)
        hf_encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        outputs = model.mt5.generate(
            encoder_outputs=hf_encoder_outputs,
            max_length=CONFIG["MAX_TARGET_LEN"]
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred_text


def run_collapse_diagnostic(model, tokenizer, real_sequence_100, device):
    """Sanity check for output collapse: feeds the model three very different
    inputs (your real captured sign, all-zeros, and random noise) and prints
    all three predictions. If they come back identical, the model is NOT
    conditioning on the landmarks -- it's emitting a fixed fallback sentence
    no matter what you sign, and no amount of correct gesturing will fix that
    at inference time (it would need retraining / more data / non-greedy
    decoding). If they differ, that's real evidence it's using your input."""
    print("\n" + "-" * 60)
    print("COLLAPSE DIAGNOSTIC")
    print("-" * 60)

    real_pred = run_inference(model, tokenizer, real_sequence_100, device)
    print(f"  Real captured sign   -> {real_pred!r}")

    zeros_seq = np.zeros_like(real_sequence_100)
    zeros_pred = run_inference(model, tokenizer, zeros_seq, device)
    print(f"  All-zeros input      -> {zeros_pred!r}")

    noise_seq = np.random.uniform(0.0, 1.0, size=real_sequence_100.shape).astype(np.float32)
    noise_pred = run_inference(model, tokenizer, noise_seq, device)
    print(f"  Random noise input   -> {noise_pred!r}")

    if real_pred == zeros_pred == noise_pred:
        print("\n  ⚠️  ALL THREE PREDICTIONS ARE IDENTICAL.")
        print("  This strongly suggests the model has collapsed to a fixed")
        print("  fallback output and is NOT conditioning on the landmarks.")
        print("  Signing correctly will not fix this at inference time.")
    else:
        print("\n  ✅ Predictions differ across inputs -- the model is at least")
        print("  reacting to what's fed in. That's necessary, but not by")
        print("  itself proof of correctness -- keep testing across your")
        print("  different sign classes to check it maps distinctly.")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sana_psl_medical_finetuned.pt",
                         help="Path to the fine-tuned .pt weights file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--auto-stop-secs", type=float, default=1.2,
                         help="Auto-stop a recording after this many seconds with "
                              "no hands detected (0 disables auto-stop).")
    parser.add_argument("--no-translate", action="store_true",
                         help="Skip the English->Urdu medical_dictionary lookup step")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.model, device)

    extractor = MediaPipeLiveExtractor(
        min_detection_conf=CONFIG["MIN_DETECTION_CONF"],
        min_tracking_conf=CONFIG["MIN_TRACKING_CONF"],
        mirror_fix=CONFIG["MIRROR_CORRECTION"],
        alpha=CONFIG["SMOOTHING_ALPHA"],
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    recording = False
    buffer = []
    last_hand_seen_time = None
    last_prediction = ""
    last_urdu = ""
    last_padded_sequence = None  # kept for the 'd' collapse diagnostic

    print("\n" + "=" * 60)
    print("Ready. Press SPACE to start recording a sign, SPACE again to stop.")
    print("Press 'm' to toggle mirror-hand-correction.")
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
                    print(f"Auto-stopped after {args.auto_stop_secs}s of no hands. "
                          f"Captured {len(buffer)} frames.")
            else:
                cv2.putText(frame, "Press SPACE to record a sign", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            raw_labels = debug_info.get("raw_labels", [])
            cv2.putText(frame, f"Raw MediaPipe labels this frame: {raw_labels}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Mirror correction: {extractor.mirror_fix}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if last_prediction:
                cv2.putText(frame, f"Prediction: {last_prediction}", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if last_urdu:
                    cv2.putText(frame, f"Urdu: {last_urdu}", (10, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("SANA A-PSL Live Inference", frame)
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

                    if len(buffer) < 3:
                        print("Too few frames captured — try again with a longer sign.")
                    else:
                        raw_seq = np.array(buffer, dtype=np.float32)
                        resampled_60 = resample_sequence(raw_seq, target_frames=CONFIG["TARGET_FRAMES"])
                        padded_100 = pad_to_max_seq_len(resampled_60, CONFIG["MAX_SEQ_LEN"], CONFIG["INPUT_DIM"])

                        last_padded_sequence = padded_100
                        pred_text = run_inference(model, tokenizer, padded_100, device)
                        last_prediction = pred_text
                        print(f"\n>>> Prediction: {pred_text}")

                        if not args.no_translate:
                            last_urdu = MEDICAL_DICTIONARY.get(pred_text, "")
                            if last_urdu:
                                print(f">>> Urdu: {last_urdu}")
                        print()

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
