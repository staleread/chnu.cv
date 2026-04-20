# Live Detection Report

Real-time test using dlib HOG+SVM detector with 68-point landmark prediction via webcam. Three distances and several occlusion/lighting conditions were tested to probe the limits of the detector.

---

## Close to camera

At close range the face fills a large portion of the frame, giving the detector the best possible signal.

Normal face — baseline, all 68 landmarks placed correctly.
<img width="1280" height="796" alt="image" src="https://github.com/user-attachments/assets/f4c6712e-990c-410f-8882-8dea9abd0050" />

Smiling — expression does not affect detection or landmark placement.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/2f9eabc6-83b2-44c4-b928-12c17c27f89c" />

Blinking — mid-blink frame still detected; landmark points follow the closed eyelid shape.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/f97799f4-f929-44dc-b027-d372e9626a1f" />

Hat with eyebrows hidden — **not detected**. The brow ridge is a strong gradient cue in HOG; covering it breaks the frontal-face template.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/6e550758-751e-446f-9f89-b2c0ced42199" />

Hat with eyebrows visible — detected normally. The hat brim alone does not interfere as long as the brow gradient is present.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/7903ac75-64b2-4e2d-9b3d-59379631d3f5" />

Swimming goggles (even lighting) — detected.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/11cdf712-675a-4685-b9d3-42c0bcac7645" />

Swimming goggles facing the window — **not detected**. The goggle lenses reflect sunlight from the window, washing out the gradient structure around the eye region. Detection only succeeded after turning the face slightly away from the window so the front of the face was no longer in the reflection.
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/ac1936d0-ecb2-4c2c-ba5c-2ec495d0a116" />

---

## 1.5 metres from camera

The face subtends fewer pixels but the detector still finds it reliably in all three conditions tested here. The hat was worn with eyebrows visible in both distance tests.

Normal
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/815f9e80-79b6-421a-9639-fdfab72ee0ee" />

Hat (eyebrows visible)
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/e3734195-018e-46c4-b282-790f34aa37a9" />

Goggles
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/573095a2-4205-42c9-9566-a43648754951" />

---

## 2–2.5 metres from camera

All three conditions detected successfully at this distance. As at 1.5 m, the hat was worn with eyebrows visible.

Normal
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/bcc89c29-882f-4f7b-b44d-ba75500ecb8f" />

Hat
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/3aee2c3d-d9b3-4dc1-893e-52d234d5c744" />

Goggles
<img width="1280" height="576" alt="image" src="https://github.com/user-attachments/assets/961d2e22-ab14-4546-a8b5-f6b2ff7f0b76" />

---

## Conclusions

| Condition | Close | 1.5 m | 2–2.5 m |
|---|---|---|---|
| Normal | ✓ | ✓ | ✓ |
| Smiling / blinking | ✓ | — | — |
| Hat (eyebrows hidden) | ✗ | ✓ | ✓ |
| Hat (eyebrows visible) | ✓ | ✓ | ✓ |
| Goggles (even light) | ✓ | ✓ | ✓ |
| Goggles (side light) | ✗ | — | — |

**Brow gradient is load-bearing.** The single most reliable way to defeat dlib's HOG+SVM detector at close range is to hide the eyebrows. The brow ridge produces a strong, consistent gradient edge that the frontal-face template depends on. Covering it causes a miss; keeping it visible restores detection even with a hat.

**Specular reflection is as disruptive as occlusion.** Goggles with no window reflection were detected fine — the eye occlusion alone did not break detection. But when the lenses reflected sunlight directly at the camera, the blown-out gradient around the eye region was enough to cause a miss. The fix was simply to angle the face away from the window.

**Landmark quality degrades gracefully.** Even in borderline cases where a detection still fires, landmark points adapt to the actual face geometry — closed eyes during a blink are traced correctly, and goggles shift the eye-region points onto the goggle frame rather than hallucinating invisible eyes.
