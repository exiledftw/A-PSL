# HANDOFF: BVH to FBX Retargeting in Three.js

**To:** Claude (Sonnet)
**From:** Antigravity (Agent)
**Project Context:** We are building a WebGL Sign Language Avatar viewer. We have a base Mixamo character (`Y Bot.fbx`) and 4 newly generated `.bvh` animation files representing sign language motions.

### The Goal
Play the `.bvh` files on the `Y Bot.fbx` mesh dynamically inside the browser.

### The Current Setup
We created `avatar_bvh_player.html`, which uses Three.js's `FBXLoader` and `BVHLoader`. You can run it via `python -m http.server 8000`.

### What We Tried (And Why It's Failing)
1. **Binding:** We successfully bound the animation tracks to the FBX bones by stripping the `mixamorig:` prefix from both the Mesh's bones and the BVH tracks.
2. **Translation Collapse:** Initially, the BVH `.position` tracks overrode the FBX bone lengths, causing the mesh to instantly shrink and collapse. We fixed this by filtering the `clip.tracks` and deleting all `.position` tracks except for the root `Hips`.
3. **The Current Bug (Twisted Joints):** Even with only `.quaternion` tracks applied, the avatar's arms and shoulders are completely twisted and mangled (see the user's latest screenshot). 

### The Root Cause
There is a **coordinate space or rest-pose mismatch** between the BVH file's generated skeleton and Mixamo's `Y Bot.fbx` skeleton. Direct quaternion injection via `AnimationMixer` fails because the BVH assumes a different base axis alignment or T-Pose than the FBX uses.

### The Ask
Please fix the twisted bone retargeting issue. You can either:
1. **Option A (Browser-side Fix):** Write a mathematical retargeting logic inside `avatar_bvh_player.html` that calculates the offset between the BVH rest pose and the FBX rest pose, applying the correct quaternion math to fix the twisted axes.
2. **Option B (Python/Blender Fix):** If browser-side retargeting is too messy, write a headless Blender Python script that permanently bakes these `.bvh` files onto the `Y Bot.fbx` skeleton and exports them as pure `.fbx` animations (similar to our previously working `23_paani.fbx`).

We leave the architectural choice to you!
