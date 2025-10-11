import cv2
import numpy as np

# === 1. Load the frame ===
frame = cv2.imread("first_frame.bmp")
if frame is None:
    raise FileNotFoundError("Could not read 'first_fram.bmp'")

frame = cv2.resize(frame, (960, 540))
height, width = frame.shape[:2]

# === 2. Color masking for white & yellow lanes ===
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
mask = cv2.bitwise_or(white_mask, yellow_mask)
masked = cv2.bitwise_and(frame, frame, mask=mask)

# === 3. Preprocess: grayscale, blur, morphology, edges ===
gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clean = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)
edges = cv2.Canny(clean, 50, 150)

# === 4. Define ROI ===
roi_mask = np.zeros_like(edges)
roi_vertices = np.array([[
    (int(0.1 * width), height),
    (int(0.9 * width), height),
    (int(0.6 * width), int(0.6 * height)),
    (int(0.4 * width), int(0.6 * height))
]], dtype=np.int32)
cv2.fillPoly(roi_mask, roi_vertices, 255)
cropped = cv2.bitwise_and(edges, roi_mask)

# === 5. Detect lines (Hough Transform) ===
lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 40,
                        minLineLength=40, maxLineGap=60)

# === 6. Separate left/right lines ===
left_lines, right_lines = [], []
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.4:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.4:
            right_lines.append((x1, y1, x2, y2))

# === 7. Average each lane ===
def average_lane(lines):
    if not lines:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs += [x1, x2]
        ys += [y1, y2]
    fit = np.polyfit(ys, xs, 1)  # x = a*y + b
    y1, y2 = height, int(height * 0.6)
    x1, x2 = int(fit[0] * y1 + fit[1]), int(fit[0] * y2 + fit[1])
    return (x1, y1, x2, y2)

left_avg = average_lane(left_lines)
right_avg = average_lane(right_lines)

# === 8. Draw lanes + center path ===
output = frame.copy()
lane_center_x = None

if left_avg is not None:
    cv2.line(output, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 8)
if right_avg is not None:
    cv2.line(output, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 8)

if left_avg and right_avg:
    mid_bottom = ((left_avg[0] + right_avg[0]) // 2, height)
    mid_top = ((left_avg[2] + right_avg[2]) // 2, int(height * 0.6))
    lane_center_x = mid_bottom[0]
    cv2.line(output, mid_bottom, mid_top, (255, 0, 0), 8)
elif left_avg:
    dx = int(width * 0.25)
    lane_center_x = left_avg[0] + dx
    cv2.line(output, (left_avg[0] + dx, height), (left_avg[2] + dx, int(height * 0.6)), (255, 0, 0), 8)
elif right_avg:
    dx = int(width * 0.25)
    lane_center_x = right_avg[0] - dx
    cv2.line(output, (right_avg[0] - dx, height), (right_avg[2] - dx, int(height * 0.6)), (255, 0, 0), 8)

# === 9. Compute steering value ===
steering_value = 0.0
steering_text = "No lane detected"

if lane_center_x is not None:
    frame_center_x = width // 2
    deviation = lane_center_x - frame_center_x
    steering_value = deviation / (width / 2)  # normalize to [-1, 1]
    steering_value = np.clip(steering_value, -1.0, 1.0)

    # Interpret steering command
    if abs(steering_value) < 0.1:
        steering_text = "Go Straight"
        color = (0, 255, 0)
    elif steering_value > 0.1:
        steering_text = f"Turn Right ({steering_value:.2f})"
        color = (0, 255, 255)
    else:
        steering_text = f"Turn Left ({steering_value:.2f})"
        color = (0, 0, 255)

    # Draw arrow
    cv2.arrowedLine(
        output,
        (frame_center_x, height),
        (int(lane_center_x), int(height * 0.8)),
        color, 5, tipLength=0.3
    )
else:
    color = (0, 0, 255)

# === 10. Annotate output ===
cv2.putText(output, steering_text, (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
cv2.putText(output, f"Steering Value: {steering_value:+.2f}", (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# === 11. Display ===
cv2.imshow("Free Way + Steering Output", output)
cv2.imwrite("first_frame_out.bmp", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: save
cv2.imwrite("free_way_with_steering.png", output)

print(f"Steering Value: {steering_value:+.2f} ({steering_text})")
