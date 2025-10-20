'''
@ 2025, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''
import avisengine
import config
import time
import cv2
import numpy as np

# === Helper: find intersection with bottom ===
def x_at_bottom(x1, y1, x2, y2, roi_h, roi_w):
    if x2 == x1:
        return x1
    m = (y2 - y1) / (x2 - x1)
    y = roi_h - 1
    x = x1 + (y - y1) / m
    return int(np.clip(x, 0, roi_w - 1))

def process(img):
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

    # Compute averaged x positions at bottom
    right_xs = [x_at_bottom(x1, y1, x2, y2, roi_h, roi_w) for (x1, y1, x2, y2, _) in right_lines]
    left_xs = [x_at_bottom(x1, y1, x2, y2, roi_h, roi_w) for (x1, y1, x2, y2, _) in left_lines]

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
    cv2.imshow("annotated_frame", out)

    return steering_norm

# Creating an instance of the Car class
car = avisengine.Car()

# Connecting to the server (Simulator)
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Counter variable
counter = 0

debug_mode = False

# Sleep for 3 seconds to make sure that client connected to the simulator
time.sleep(3)

try:
    while(True):
        # Counting the loops
        counter = counter + 1

        # Set the power of the engine the car to 20, Negative number for reverse move, Range [-100,100]
        car.setSpeed(10)

        # Set the angle between sensor rays to 45 degrees, Use this only if you want to set it from python client
        # Notice: Once it is set from the client, it cannot be changed using the GUI
        car.setSensorAngle(45)

        # Get the data. Need to call it every time getting image and sensor data
        car.getData()

        # Display the FPS on the frame
        # Start getting image and sensor data after 4 loops
        if(counter > 4):

            # Returns a list with three items which the 1st one is Left sensor data\
            # the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors()

            # Returns an opencv image type array. if you use PIL you need to invert the color channels.
            image = car.getImage()

            value = process(image)

            car.setSteering(10 * value)

            if(debug_mode):
                # Returns an integer which is the real time car speed in KMH
                carSpeed = car.getSpeed()
                print(f"Speed : {carSpeed}")
                print(f'Left : {str(sensors[0])} | Middle : {str(sensors[1])} | Right : {str(sensors[2])}')
                if image is not None and image.any():
                    # Showing the opencv type image
                    cv2.imshow('frames', image)

            if cv2.waitKey(10) == ord('q'):
                break

finally:
    car.stop()
