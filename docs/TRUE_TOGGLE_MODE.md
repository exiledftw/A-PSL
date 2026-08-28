# True Toggle Mode Implementation
- Fixed regression where spacebar still required holding
- Implemented `ctypes.windll.user32.GetAsyncKeyState(0x20)` edge detection (transition from up to down) to perfectly toggle recording state
- Users can now safely press space once, perform two-handed signs freely, and press space again to translate