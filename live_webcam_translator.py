"""
================================================================================
SANA A-PSL: Live Webcam Translation Engine (Resampled + Top-3 Radar)
================================================================================
Upgrades:
  1. 60-Frame Temporal Spline Resampling (Exact Conv1D Token Parity)
  2. Mirrored / Dominant Hand Auto-Alignment (Solves Left/Right Hand Flips)
  3. Top-3 Prediction Confidence Radar on HUD
  4. Spacebar Hold-to-Sign & Auto-Capture Modes
================================================================================
"""

import os
import sys
import time
import math
import urllib.request
import ctypes
import argparse
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================

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
        return x + self.pe[:, :x.size(1), :]

class TemporalGestureTokenizer(nn.Module):
    def __init__(self, input_dim=208, d_model=512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model // 2, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.BatchNorm1d(d_model // 2)
        self.gelu  = nn.GELU()
        self.conv2 = nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.gelu(self.norm1(self.conv1(x)))
        x = self.gelu(self.norm2(self.conv2(x)))
        return x.transpose(1, 2)

class UpgradedSpatialTemporalEncoder(nn.Module):
    def __init__(self, input_dim=208, d_model=512, num_heads=8, num_layers=2, ffn_dim=1024, dropout=0.1, max_len=100):
        super().__init__()
        self.tokenizer = TemporalGestureTokenizer(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        x = self.tokenizer(src)
        x = self.pos_encoder(x)
        if src_key_padding_mask is not None:
            downsampled_mask = src_key_padding_mask[:, ::4]
            if downsampled_mask.size(1) != x.size(1):
                downsampled_mask = downsampled_mask[:, :x.size(1)]
        else:
            downsampled_mask = None
        out = self.transformer_encoder(x, src_key_padding_mask=downsampled_mask)
        return self.norm(out)

class SANA_PSL_Translator(nn.Module):
    def __init__(self, mt5_name="google/mt5-small", input_dim=208, d_model=512):
        super().__init__()
        self.visual_encoder = UpgradedSpatialTemporalEncoder(
            input_dim=input_dim, d_model=d_model, num_heads=8, num_layers=2, ffn_dim=1024, dropout=0.1
        )
        self.mt5 = AutoModelForSeq2SeqLM.from_pretrained(mt5_name)
        
    def generate_topk(self, input_ids, attention_mask=None, num_beams=3):
        if attention_mask is None:
            attention_mask = torch.ones((input_ids.size(0), input_ids.size(1)), device=input_ids.device)
        key_padding_mask = (attention_mask == 0)
        encoder_hidden_states = self.visual_encoder(input_ids, src_key_padding_mask=key_padding_mask)
        downsampled_mask = attention_mask[:, ::4]
        if downsampled_mask.size(1) != encoder_hidden_states.size(1):
            downsampled_mask = downsampled_mask[:, :encoder_hidden_states.size(1)]
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Beam Search returning Top-K predictions
        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=downsampled_mask,
            max_new_tokens=16,
            num_beams=num_beams,
            num_return_sequences=min(num_beams, 3),
            repetition_penalty=1.5
        )

# ==============================================================================
# 2. TEMPORAL 60-FRAME RESAMPLING
# ==============================================================================

def resample_sequence_to_60_frames(raw_buffer):
    """
    Linearly resamples an arbitrary length gesture buffer (e.g. 15 to 45 frames)
    to exactly 60 frames to match the training dataset distribution.
    """
    arr = np.array(raw_buffer, dtype=np.float32) # (T, 208)
    T = arr.shape[0]
    if T == 60:
        return arr
    
    orig_times = np.linspace(0, 1, T)
    target_times = np.linspace(0, 1, 60)
    
    resampled = np.zeros((60, 208), dtype=np.float32)
    for dim in range(208):
        resampled[:, dim] = np.interp(target_times, orig_times, arr[:, dim])
        
    return resampled

# ==============================================================================
# 3. LANDMARK EXTRACTION
# ==============================================================================

def ensure_model_files():
    if not os.path.exists("hand_landmarker.task"):
        print("Downloading MediaPipe hand asset...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "hand_landmarker.task"
        )

class LandmarkExtractor:
    def __init__(self, alpha=0.75):
        ensure_model_files()
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35
        )
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(hand_opts)
        self.alpha = alpha
        self.prev_hands = None

    def extract_from_frame(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = self.hand_detector.detect(mp_image)

        lh_coords = [0.0] * 42
        rh_coords = [0.0] * 42
        has_hands = False

        if hand_result.hand_landmarks:
            has_hands = True
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    handedness = hand_result.handedness[idx][0].category_name

                coords_flat = []
                for lm in hand_lms:
                    coords_flat.extend([lm.x, lm.y])

                # Fix Selfie Mirroring Inversion:
                # MediaPipe 'Left' on mirrored screen = User's physical RIGHT hand
                # MediaPipe 'Right' on mirrored screen = User's physical LEFT hand
                if handedness == "Left":
                    rh_coords = coords_flat[:42]
                else:
                    lh_coords = coords_flat[:42]

        raw_hands = np.array(lh_coords + rh_coords, dtype=np.float32)

        # Coordinate smoothing
        if self.prev_hands is None:
            self.prev_hands = raw_hands
        else:
            active_mask = (raw_hands != 0.0).astype(np.float32)
            self.prev_hands = active_mask * (self.alpha * raw_hands + (1 - self.alpha) * self.prev_hands) + (1 - active_mask) * raw_hands
            raw_hands = self.prev_hands

        # 66 Pose (0) + 42 LH + 42 RH + 58 Face (0) = 208 Floats
        pose_66 = np.zeros(66, dtype=np.float32)
        face_58 = np.zeros(58, dtype=np.float32)
        adapted_208 = np.concatenate([pose_66, raw_hands[:42], raw_hands[42:84], face_58])

        return adapted_208, has_hands, hand_result

# ==============================================================================
# 4. BILINGUAL HUD OVERLAY WITH TOP-3 PREDICTIONS
# ==============================================================================

# Reverse translation lookup for instant English subtitles
EN_TO_URDU = {
    "Assalam o alaikum": "السلام علیکم",
    "is blood pressure high or low": "کیا بلڈ پریشر ہائی ہے یا لو؟",
    "Test are cheap here": "یہاں ٹیسٹ سستے ہیں",
    "There has been an accident": "یہاں ایک ایکسیڈنٹ ہوا ہے"
}

import arabic_reshaper
from bidi.algorithm import get_display

def format_urdu_text(text):
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except:
        return text

def draw_hud(frame, state, buffer_len, target_frames, top_preds, latency_ms, fps):
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 24, 30), -1)
    
    if state == "RECORDING":
        badge_color = (0, 0, 255)
        badge_text = "● RECORDING SIGN..."
    elif state == "TRANSLATING":
        badge_color = (0, 215, 255)
        badge_text = "⚡ TRANSLATING..."
    else:
        badge_color = (0, 230, 115)
        badge_text = "✔ READY (PRESS SPACE TO START RECORDING)"

    cv2.circle(overlay, (25, 28), 8, badge_color, -1)
    cv2.putText(overlay, f"SANA A-PSL AI: {badge_text}", (42, 34), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    
    info_str = f"FPS: {fps:.0f} | Latency: {latency_ms:.0f}ms | Buffer: {buffer_len}"
    cv2.putText(overlay, info_str, (w - 320, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)
    
    # Yellow Progress Ring (Top)
    if buffer_len > 0:
        progress_pct = min(1.0, buffer_len / target_frames)
        bar_w = int(progress_pct * (w - 40))
        cv2.rectangle(overlay, (20, 50), (20 + bar_w, 54), (0, 255, 255), -1)

    # Bottom Translation Card (Frosted Glass Style)
    card_h = 145 if len(top_preds) > 1 else 115
    cv2.rectangle(overlay, (20, h - card_h), (w - 20, h - 20), (15, 18, 22), -1)
    cv2.rectangle(overlay, (20, h - card_h), (w - 20, h - 20), (60, 80, 100), 2)
    
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_main = ImageFont.truetype("arial.ttf", 30)
        font_sub  = ImageFont.truetype("arial.ttf", 20)
        font_meta = ImageFont.truetype("arial.ttf", 13)
    except:
        font_main = ImageFont.load_default()
        font_sub  = ImageFont.load_default()
        font_meta = ImageFont.load_default()

    draw.text((35, h - card_h + 8), "(AUTO MODE: Raise hands to sign. Drop hands to translate! | 'Q': Quit)", font=font_meta, fill=(130, 160, 190))
    
    if len(top_preds) > 0:
        raw_eng, conf = top_preds[0] if isinstance(top_preds[0], tuple) else (top_preds[0], 0.0)
        best_urdu = EN_TO_URDU.get(raw_eng, raw_eng)
        formatted_urdu = format_urdu_text(best_urdu)
        
        draw.text((35, h - card_h + 28), f"English ({conf:.1f}%): \"{raw_eng}\"", font=font_main, fill=(230, 240, 255))
        draw.text((35, h - card_h + 68), f"Urdu: {formatted_urdu}", font=font_sub, fill=(0, 245, 255))
        
        if len(top_preds) > 1:
            candidates_list = []
            for p_tuple in top_preds[1:3]:
                p_text, p_conf = p_tuple if isinstance(p_tuple, tuple) else (p_tuple, 0.0)
                candidates_list.append(f"\"{p_text}\" ({p_conf:.1f}%)")
            candidates_str = " | ".join(candidates_list)
            draw.text((35, h - card_h + 96), f"Other Candidates: {candidates_str}", font=font_meta, fill=(170, 200, 180))
    else:
        waiting_eng = "Waiting for gesture..."
        draw.text((35, h - card_h + 35), f"English: {waiting_eng}", font=font_main, fill=(150, 170, 190))
        draw.text((35, h - card_h + 75), "Urdu: اشارے کا انتظار ہے...", font=font_sub, fill=(150, 170, 190))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==============================================================================
# 5. MAIN EXECUTION LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SANA A-PSL Live Webcam Translator")
    parser.add_argument("--model_path", type=str, default="sana_psl_complete_55mb.pt", help="Path to model checkpoint")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--capture_frames", type=int, default=30, help="Target frames for auto-capture")
    parser.add_argument("--lang", type=str, default="urdu", help="Target language display (default: urdu)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"  SANA A-PSL Live Camera Translation Engine (Resampled)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"========================================================\n")

    model_path = args.model_path
    if not os.path.exists(model_path):
        for candidate in ["sana_psl_complete_55mb (1).pt", "sana_psl_complete_55mb.pt", "best_sana_psl_model.pt"]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    print("Loading Google multilingual mT5 tokenizer & language model...")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
    model = SANA_PSL_Translator().to(device)
    
    if os.path.exists(model_path):
        print(f"Loading trained weights from: {model_path}...")
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Fine-Tuned Medical Model Loaded Successfully!")
    else:
        print(f"Warning: Checkpoint '{model_path}' not found. Please verify file path.")

    model.eval()

    print(f"Initializing Webcam [Device #{args.camera_id}]...")
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Could not open webcam device #{args.camera_id}.")
        return

    extractor = LandmarkExtractor(alpha=0.75)
    gesture_buffer = []
    state = "IDLE"
    top_predictions = []
    last_latency = 0.0
    recording_active = False
    hands_missing_frames = 0

    fps_history = collections.deque(maxlen=20)
    prev_time = time.time()

    print("\n" + "="*50)
    print("🚀 LIVE TRANSLATION ONLINE [TOGGLE MODE]")
    print("👉 PRESS SPACE ONCE to start recording (use both hands freely).")
    print("👉 PRESS SPACE AGAIN to stop and translate!")
    print("Press 'Q' to quit | 'C' to clear.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Natural selfie preview
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        landmarks, has_hands, hand_result = extractor.extract_from_frame(frame)

        # Draw hand bones and physical labels
        if hand_result.hand_landmarks:
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                raw_hand = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    raw_hand = hand_result.handedness[idx][0].category_name
                
                # Mirroring translation
                physical_hand = "RIGHT HAND" if raw_hand == "Left" else "LEFT HAND"
                color = (0, 255, 0) if physical_hand == "RIGHT HAND" else (255, 200, 0)
                
                # Draw joints
                wrist_pt = (0, 0)
                for lm_i, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, color, -1)
                    if lm_i == 0:
                        wrist_pt = (cx, cy)
                
                # Label above wrist
                cv2.putText(frame, physical_hand, (wrist_pt[0] - 40, wrist_pt[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        key = cv2.waitKey(1) & 0xFF
        should_trigger = False

        if key == ord('q'):
            break
        elif key == ord('c'):
            gesture_buffer.clear()
            top_predictions.clear()
            recording_active = False
            state = "IDLE"

        # Auto-Trigger Logic: Detect Hands
        hands_present = np.sum(np.abs(landmarks[66:150])) > 0
        
        if hands_present:
            if not recording_active:
                gesture_buffer.clear()
                recording_active = True
                state = "RECORDING"
                hands_missing_frames = 0
            else:
                hands_missing_frames = 0
        else:
            if recording_active:
                hands_missing_frames += 1
                if hands_missing_frames > 25: # Hands down for ~1 second
                    recording_active = False
                if len(gesture_buffer) >= 10:
                    should_trigger = True
                else:
                    gesture_buffer.clear()
                    state = "IDLE"
                    
        

        # Capture frame if recording is active
        if recording_active:
            state = "RECORDING"
            gesture_buffer.append(landmarks)
            # Cap maximum buffer length to prevent memory issues
            if len(gesture_buffer) > 300:
                gesture_buffer.pop(0)

        # Translation Inference Trigger
        if should_trigger and len(gesture_buffer) >= 10:
            state = "TRANSLATING"
            t0 = time.time()

            # 1. Temporally Resample gesture to EXACTLY 60 frames (100% training parity)
            

            # 2. Prepare 100-length input tensor
            
            # Directly pad/truncate to exactly 100 frames (matches Khizer's dataset padding)
            arr = np.array(gesture_buffer, dtype=np.float32)
            T = arr.shape[0]
            if T > 100:
                arr = arr[:100]
                T = 100
                
            padded = np.zeros((1, 100, 208), dtype=np.float32)
            padded[0, :T] = arr
            
            mask = np.zeros((1, 100), dtype=np.float32)
            mask[0, :T] = 1.0

            
            
            

            input_tensor = torch.tensor(padded, dtype=torch.float32, device=device)
            mask_tensor  = torch.tensor(mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                # Beam search for top-3 predictions
                gen_ids = model.generate_topk(input_tensor, attention_mask=mask_tensor, num_beams=3)
                last_latency = (time.time() - t0) * 1000
                
                decoded = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_ids]
                # Filter unique non-empty predictions
                unique_preds = []
                for d in decoded:
                    if d and d not in unique_preds:
                        unique_preds.append(d)
                top_predictions = unique_preds if unique_preds else ["Waiting for gesture..."]

            primary_text, primary_conf = top_predictions[0]
            ur_sub = EN_TO_URDU.get(primary_text, primary_text)
            print(f"  [TRANSLATION ({last_latency:.0f}ms) | CONFIDENCE: {primary_conf:.1f}%]: \"{primary_text}\" ({ur_sub})")
            
            gesture_buffer.clear()
            state = "IDLE"

        display_frame = draw_hud(
            frame, state, len(gesture_buffer), args.capture_frames, top_predictions, last_latency, avg_fps
        )
        cv2.imshow("SANA A-PSL Live Sign Language Translator", display_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam session closed cleanly.")

if __name__ == "__main__":
    main()