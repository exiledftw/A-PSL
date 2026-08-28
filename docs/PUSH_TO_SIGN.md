# Strict Push-To-Sign Mode
- Replaced auto-capture with strict Spacebar requirement using Windows `ctypes.windll.user32.GetAsyncKeyState`
- Fixed idle background noise generation where resting hands produced false positives like 'garden' and 'goodbye'
- Updated HUD to reflect strict spacebar mode
