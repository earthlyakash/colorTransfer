
# Skin Tone Transfer using Python

This project allows you to transfer the **skin tone** from a source image to a target image using classical computer vision techniques. It uses **OpenCV, NumPy, Tkinter**, and **Matplotlib**.

## 🔥 Features

- Detect skin regions in both source and target images using color space rules (HSV, YCrCb, RGB)
- Match the skin color from the source to the target while preserving the original structure
- Merge recolored skin back into the target image
- Visual side-by-side comparison using `matplotlib`
- GUI-based image selection via Tkinter

---

## 📦 Dependencies

Install via pip:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python
numpy
matplotlib
tk
```

---

## 🚀 How to Run

1. Clone the repository or copy the script.
2. Run the script using Python 3:

```bash
python matchColor_V2.py
```

3. You will be prompted to choose:
   - A **source image** (for the skin tone)
   - A **target image** (to apply the tone to)
4. The result will be saved as `*_skin_recolored.png` in the same folder as your target.

---

## 🖼 Example Workflow

| Source Image (Skin Tone) | Target Image | Result with Transferred Skin |
|--------------------------|--------------|-------------------------------|
| ![source](examples/source.jpg) | ![target](examples/target.jpg) | ![result](examples/result.jpg) |

---

## 📄 License

FREE

---

## ✨ Credits

Developed using OpenCV and NumPy by [Akash Kumar].
