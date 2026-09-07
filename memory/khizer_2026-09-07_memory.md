# Khizer / Rehan - 2026-09-07

## XR Animator Pipeline & NLP Audio Trigger Built
- Abandoned the custom Three.js/WebGL HTML player approach in favor of a robust VTubing architecture using XR Animator directly.
- **Pipeline:** Speech -> audio_listener.py (SpeechRecognition + thefuzz NLP) -> UDP Socket (Port 5005) -> stream_controller.py (OpenCV) -> OBS Virtual Camera -> XR Animator -> SANA Avatar.
- stream_controller.py dynamically handles switching the virtual camera feed between an idle.png state and human reference gesture videos (.mp4).
- Claude successfully generated the scripts for both the UDP video controller and the fuzzy-matching audio listener.
- Files have been organized into the xr_pipeline/ directory in the main repo.
- The next dependency is waiting for Khizer to record the full 40+40 phrase medical dataset videos so they can be dropped into the ideos/ folder.
