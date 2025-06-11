import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, messagebox
import matplotlib.pyplot as plt

# Select file with prompt
def select_file(prompt="Select an image"):
    Tk().withdraw()
    print(prompt)
    return filedialog.askopenfilename(
        title=prompt,
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

# Extract skin using HSV, YCrCb, and RGB rules
def extract_skin(image):
    if image is None:
        return None, None

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower_hsv = np.array([0, 40, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    mask_rgb = np.logical_and.reduce((
        r > 95, g > 40, b > 20,
        (r - g) > 15,
        r > b,
        np.abs(r - g) > 15
    )).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    combined_mask = cv2.bitwise_and(combined_mask, mask_rgb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

    skin_only = cv2.bitwise_and(image, image, mask=combined_mask)
    return combined_mask, skin_only

# Compute mean & std over mask
def masked_mean_std(image, mask):
    pixels = image[mask > 0]
    if pixels.size == 0:
        return np.zeros(3), np.ones(3)  # Prevent divide-by-zero
    return np.mean(pixels, axis=0), np.std(pixels, axis=0)

# Step 1: Choose source and target
source_path = select_file("Select Source Image (skin tone)")
target_path = select_file("Select Target Image (to apply skin tone)")

source_img = cv2.imread(source_path)
target_img = cv2.imread(target_path)

if source_img is None or target_img is None:
    messagebox.showerror("Error", "One or both images could not be loaded.")
    exit()

# Resize source to match target
source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))

# Step 2: Extract skin masks and regions
src_mask, src_skin = extract_skin(source_img)
tgt_mask, tgt_skin = extract_skin(target_img)

if src_skin is None or tgt_skin is None:
    messagebox.showerror("Error", "Skin extraction failed.")
    exit()

# Step 3: Compute mean/std for recoloring
src_mean, src_std = masked_mean_std(src_skin.astype(np.float32), src_mask)
tgt_mean, tgt_std = masked_mean_std(tgt_skin.astype(np.float32), tgt_mask)

# Step 4: Match source skin tone to target
matched_skin = tgt_skin.astype(np.float32)
for c in range(3):
    matched_skin[..., c] = ((matched_skin[..., c] - tgt_mean[c]) *
                            (src_std[c] / (tgt_std[c] + 1e-6)) + src_mean[c])

matched_skin = np.clip(matched_skin, 0, 255).astype(np.uint8)

# Step 5: Merge matched skin into original target
final_result = target_img.copy()
for c in range(3):
    final_result[:, :, c] = np.where(tgt_mask > 0, matched_skin[:, :, c], target_img[:, :, c])

# Step 6: Save and show result
output_path = os.path.splitext(target_path)[0] + "_skin_recolored.png"
cv2.imwrite(output_path, final_result)
messagebox.showinfo("Done", f"Recolored skin merged with original target.\nSaved as:\n{output_path}")

# Optional: Show visual comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Source Image (Skin Color)")
plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Original Target")
plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Final Result (Recolored Skin)")
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
