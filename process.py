import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load and prepare ===
img = cv2.imread("first_frame.bmp")
if img is None:
    raise FileNotFoundError("Could not open first_frame.bmp")

h, w = img.shape[:2]

# Use only bottom half
roi = img[h//2:h, :].copy()
roi_h, roi_w = roi.shape[:2]

# === Preprocess ===
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# === Detect line segments ===
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                        minLineLength=40, maxLineGap=40)

left_lines, right_lines, vertical_edges = [], [], []

if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Classify lines by slope
        if slope == np.inf or abs(slope) > 5:
            vertical_edges.append((x1, y1, x2, y2))  # sidewalk/curb
        elif slope < -0.3:
            left_lines.append((x1, y1, x2, y2, slope))
        elif slope > 0.3:
            right_lines.append((x1, y1, x2, y2, slope))
        else:
            # shallow slopes near sides → possible sidewalk edge
            if x1 < roi_w*0.25 or x2 < roi_w*0.25 or x1 > roi_w*0.75 or x2 > roi_w*0.75:
                vertical_edges.append((x1, y1, x2, y2))

# === Helper: find intersection with bottom ===
def x_at_bottom(x1, y1, x2, y2):
    if x2 == x1:
        return x1
    m = (y2 - y1) / (x2 - x1)
    y = roi_h - 1
    x = x1 + (y - y1) / m
    return int(np.clip(x, 0, roi_w - 1))

# Compute averaged x positions at bottom
right_xs = [x_at_bottom(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in right_lines]
left_xs = [x_at_bottom(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in left_lines]

# === Estimate lane center ===
if right_xs and left_xs:
    right_bot = int(np.mean(right_xs))
    left_bot = int(np.mean(left_xs))
    lane_center_x = (right_bot + left_bot) // 2
    note = "both lanes detected"
elif right_xs:
    right_bot = int(np.mean(right_xs))
    left_bot = right_bot - int(roi_w * 0.25)
    lane_center_x = (right_bot + max(0, left_bot)) // 2
    note = "only right lane detected; left estimated"
elif left_xs:
    left_bot = int(np.mean(left_xs))
    right_bot = left_bot + int(roi_w * 0.25)
    lane_center_x = (left_bot + min(roi_w - 1, right_bot)) // 2
    note = "only left lane detected; right estimated"
else:
    lane_center_x = roi_w * 3 // 4 - roi_w // 8
    note = "no clear lane lines; fallback used"

# === Compute steering ===
image_center_x = roi_w // 2
steering_pixels = int(lane_center_x - image_center_x)
steering_norm = steering_pixels / (roi_w / 2)
steering_norm = max(-1.0, min(1.0, steering_norm))  # clamp

# === Annotate ===
annot = roi.copy()

# Draw detected lines
for (x1, y1, x2, y2, _) in left_lines:
    cv2.line(annot, (x1, y1), (x2, y2), (0, 200, 0), 3)   # green = left lane
for (x1, y1, x2, y2, _) in right_lines:
    cv2.line(annot, (x1, y1), (x2, y2), (0, 0, 200), 3)   # blue = right lane
for (x1, y1, x2, y2) in vertical_edges:
    cv2.line(annot, (x1, y1), (x2, y2), (0, 200, 200), 2) # cyan = sidewalk

# Draw centers
cv2.line(annot, (image_center_x, 0), (image_center_x, roi_h-1), (255, 0, 255), 2)  # magenta
cv2.line(annot, (int(lane_center_x), 0), (int(lane_center_x), roi_h-1), (0, 0, 0), 3)
cv2.line(annot, (int(lane_center_x), 0), (int(lane_center_x), roi_h-1), (0, 0, 255), 2)

# Add info text
text1 = f"Steering px: {steering_pixels}, norm: {steering_norm:.3f}"
cv2.putText(annot, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(annot, note, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

# Replace ROI in original frame
out = img.copy()
out[h//2:h, :] = annot

# Save and/or display
cv2.imwrite("annotated_frame.png", out)

# Optional: show
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Annotated frame - lanes and sidewalk edges")
plt.show()

print(f"Steering pixels: {steering_pixels}, normalized: {steering_norm:.3f}")
cv2.waitKey(0)
cv2.destroyAllWindows()
