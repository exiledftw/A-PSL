import cv2
import time
import os
import ctypes
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import bidi.algorithm
import arabic_reshaper

# ==============================================================================
# 1. THE SIMPACT 2026 MEDICAL EMERGENCY PHRASES
# ==============================================================================
MEDICAL_PHRASES = [
    ("Chest_Pain", "سینے میں درد", "Chest pain"),
    ("Breathing_Problem", "سانس لینے میں مسئلہ", "Breathing problem"),
    ("Dizzy", "چکر آنا", "Dizzy / Fainting"),
    ("Vomiting", "الٹی", "Vomiting"),
    ("Bleeding", "خون بہنا", "Bleeding"),
    ("Fever", "بخار", "Fever"),
    ("Headache", "سر درد", "Headache"),
    ("Stomach_Ache", "پیٹ میں درد", "Stomach ache"),
    ("Broken_Bone", "ہڈی ٹوٹنا", "Broken bone"),
    ("Burn", "جل جانا", "Burn"),
    ("Pregnant", "حاملہ", "Pregnant"),
    ("Diabetes", "شوگر کی بیماری", "Diabetes / Sugar"),
    ("Blood_Pressure", "بلڈ پریشر", "Blood Pressure"),
    ("Heart_Attack", "دل کا دورہ", "Heart attack"),
    ("Allergy", "الرجی", "Allergy"),
    ("Medicine", "دوا", "Medicine"),
    ("Injection", "انجکشن", "Injection"),
    ("X_Ray", "ایکسرے", "X-Ray"),
    ("Blood_Test", "خون کا ٹیسٹ", "Blood test"),
    ("Where_Is_Doctor", "ڈاکٹر کہاں ہے؟", "Where is the doctor?"),
    ("Help_Me", "میری مدد کریں", "Help me"),
    ("Call_Ambulance", "ایمبولینس بلائیں", "Call ambulance"),
    ("Need_Water", "پانی چاہیے", "Need water"),
    ("Feeling_Cold", "ٹھنڈ لگ رہی ہے", "Feeling cold"),
    ("Where_Is_Pain", "درد کہاں ہے؟", "Where is the pain?"),
    ("Since_When", "کب سے؟", "Since when?"),
    ("Better", "پہلے سے بہتر", "Better than before"),
    ("Hospital", "ہسپتال", "Hospital"),
    ("Unconscious", "بے ہوش", "Unconscious"),
    ("Rest", "آرام کریں", "Rest / Relax")
]

# ==============================================================================
# 2. LANDMARK EXTRACTOR (IDENTICAL TO LIVE CAMERA FOR ZERO DOMAIN SHIFT)
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

        if hand_result.hand_landmarks:
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    handedness = hand_result.handedness[idx][0].category_name

                coords_flat = []
                for lm in hand_lms:
                    coords_flat.extend([lm.x, lm.y])

                # Fixed selfie mirror mapping
                if handedness == "Left":
                    rh_coords = coords_flat[:42]
                else:
                    lh_coords = coords_flat[:42]

        raw_hands = np.array(lh_coords + rh_coords, dtype=np.float32)

        if self.prev_hands is None:
            self.prev_hands = raw_hands
        else:
            active_mask = (raw_hands != 0.0).astype(np.float32)
            self.prev_hands = active_mask * (self.alpha * raw_hands + (1 - self.alpha) * self.prev_hands) + (1 - active_mask) * raw_hands
            raw_hands = self.prev_hands

        pose_66 = np.zeros(66, dtype=np.float32)
        face_58 = np.zeros(58, dtype=np.float32)
        adapted_208 = np.concatenate([pose_66, raw_hands[:42], raw_hands[42:84], face_58])

        return adapted_208, hand_result

# ==============================================================================
# 3. UI HELPER
# ==============================================================================
def format_urdu_text(text):
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return bidi.algorithm.get_display(reshaped_text)
    except:
        return text

def draw_hud(frame, current_idx, is_recording, takes_count):
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # Top Status Bar
    bar_color = (0, 0, 255) if is_recording else (20, 24, 30)
    cv2.rectangle(overlay, (0, 0), (w, 55), bar_color, -1)
    
    status_text = "● RECORDING..." if is_recording else "✔ READY (Press SPACE to start)"
    cv2.putText(overlay, f"DATASET RECORDER: {status_text}", (20, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                
    cv2.putText(overlay, f"Takes Saved: {takes_count}", (w - 200, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    # Bottom Prompt Card
    card_h = 130
    cv2.rectangle(overlay, (20, h - card_h), (w - 20, h - 20), (15, 18, 22), -1)
    cv2.rectangle(overlay, (20, h - card_h), (w - 20, h - 20), (60, 80, 100), 2)
    
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw text via PIL for Urdu
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font_main = ImageFont.truetype("arial.ttf", 36)
        font_sub  = ImageFont.truetype("arial.ttf", 22)
    except:
        font_main = ImageFont.load_default()
        font_sub  = ImageFont.load_default()

    eng_id, urdu_text, eng_text = MEDICAL_PHRASES[current_idx]
    
    draw.text((35, h - card_h + 10), f"Sign {current_idx+1}/{len(MEDICAL_PHRASES)}: {eng_text}", font=font_main, fill=(255, 255, 255))
    draw.text((35, h - card_h + 55), format_urdu_text(urdu_text), font=font_main, fill=(0, 255, 150))
    
    draw.text((35, h - card_h + 95), "Controls: [SPACE] Toggle Record | [N] Next Sign | [B] Back | [Q] Quit", font=font_sub, fill=(150, 150, 150))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==============================================================================
# 4. MAIN RECORDER LOOP
# ==============================================================================
def main():
    os.makedirs("medical_dataset", exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    extractor = LandmarkExtractor(alpha=0.75)
    
    current_idx = 0
    recording_active = False
    prev_space_down = False
    gesture_buffer = []

    print("\n" + "="*50)
    print("🏥 SIMPACT 2026 MEDICAL DATASET RECORDER")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        eng_id, urdu_text, eng_text = MEDICAL_PHRASES[current_idx]
        save_dir = os.path.join("medical_dataset", eng_id)
        os.makedirs(save_dir, exist_ok=True)
        takes_count = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
        
        landmarks, hand_result = extractor.extract_from_frame(frame)
        
        # Draw skeleton
        if hand_result.hand_landmarks:
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                for lm in hand_lms:
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255) if recording_active else (0, 255, 0), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('n'): 
            current_idx = min(current_idx + 1, len(MEDICAL_PHRASES) - 1)
            recording_active = False
        elif key == ord('b'): 
            current_idx = max(current_idx - 1, 0)
            recording_active = False

        # Toggle Record Logic
        is_space_down = (ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000) != 0
        
        if is_space_down and not prev_space_down:
            if not recording_active:
                gesture_buffer.clear()
                recording_active = True
                print(f"🔴 Started recording: {eng_id}")
            else:
                recording_active = False
                if len(gesture_buffer) > 5:
                    file_path = os.path.join(save_dir, f"{takes_count+1:03d}.npy")
                    np.save(file_path, np.array(gesture_buffer, dtype=np.float32))
                    print(f"✅ Saved {len(gesture_buffer)} frames to {file_path}")
                gesture_buffer.clear()
        
        prev_space_down = is_space_down

        if recording_active:
            gesture_buffer.append(landmarks)

        display_frame = draw_hud(frame, current_idx, recording_active, takes_count)
        cv2.imshow("SIMPACT 2026 Dataset Recorder", display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
