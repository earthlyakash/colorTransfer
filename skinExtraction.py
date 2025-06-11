import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, messagebox

# ✅ Open file dialog
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an image to extract skin",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
)

if not file_path:
    messagebox.showinfo("No File Selected", "You didn't choose any file. Exiting.")
    exit()

# ✅ Read image
image = cv2.imread(file_path)
if image is None:
    messagebox.showerror("Error", "Failed to load the selected image.")
    exit()

# ✅ Convert to multiple color spaces
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# ✅ Create HSV mask
lower_hsv = np.array([0, 40, 60], dtype=np.uint8)
upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

# ✅ Create YCrCb mask
lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

# ✅ Create RGB mask
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = img_rgb[:, :, 0]
g = img_rgb[:, :, 1]
b = img_rgb[:, :, 2]
mask_rgb = np.logical_and.reduce((
    r > 95, g > 40, b > 20,
    (r - g) > 15,
    r > b,
    np.abs(r - g) > 15
)).astype(np.uint8) * 255

# ✅ Combine all masks
combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
combined_mask = cv2.bitwise_and(combined_mask, mask_rgb)

# ✅ Apply morphology to clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

# ✅ Create transparent image
b, g, r = cv2.split(image)
output = cv2.merge([b, g, r, combined_mask])

# ✅ Save output
output_path = os.path.join(os.path.dirname(file_path), "skin_output_improved.png")
cv2.imwrite(output_path, output)
messagebox.showinfo("Done", f"Skin extracted and saved as:\n{output_path}")
