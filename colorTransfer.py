import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def select_file(prompt="Select an image"):
    Tk().withdraw()  # Hide tkinter window
    print(prompt)
    return filedialog.askopenfilename()

# Select source and target images
source_path = select_file("Select source image (to recolor)")
target_path = select_file("Select target image (to match colors from)")

# Load images with unchanged flag (preserves alpha if present)
source_img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
target_img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

if source_img is None or target_img is None:
    raise ValueError("One or both images could not be loaded.")

# Resize target to match source size
target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))

def process_image(img):
    if img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        rgb = img
        alpha = None
        mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)
    return rgb, alpha, mask

# Split images
src_rgb, src_alpha, src_mask = process_image(source_img)
tgt_rgb, tgt_alpha, tgt_mask = process_image(target_img)
combined_mask = src_mask & tgt_mask

# Compute masked mean and std
def masked_mean_std(image, mask):
    pixels = image[mask]
    return np.mean(pixels, axis=0), np.std(pixels, axis=0)

src_mean, src_std = masked_mean_std(src_rgb, combined_mask)
tgt_mean, tgt_std = masked_mean_std(tgt_rgb, combined_mask)

# Apply color matching
matched_rgb = src_rgb.astype(np.float32)
for c in range(3):
    matched_rgb[..., c] = ((matched_rgb[..., c] - src_mean[c]) * 
                           (tgt_std[c] / (src_std[c] + 1e-6)) + tgt_mean[c])

matched_rgb = np.clip(matched_rgb, 0, 255).astype(np.uint8)

# Merge alpha if exists
if src_alpha is not None:
    matched_img = cv2.merge([matched_rgb, src_alpha])
    display_img = cv2.cvtColor(matched_img, cv2.COLOR_BGRA2RGBA)
    disp_src = cv2.cvtColor(source_img, cv2.COLOR_BGRA2RGBA)
    disp_tgt = cv2.cvtColor(target_img, cv2.COLOR_BGRA2RGBA)
else:
    matched_img = matched_rgb
    display_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    disp_src = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    disp_tgt = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)


# Save result
output_name = os.path.splitext(source_path)[0] + '_matched.png'
cv2.imwrite(output_name, matched_img)
print(f"✅ Saved matched image: {output_name}")
