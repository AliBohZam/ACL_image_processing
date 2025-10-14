'''
@ 2025, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''
import avisengine
import config
import time
import cv2
import numpy as np

def average_lane(lines, height):
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

def process(image):
    frame = cv2.resize(image, (960, 540))
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

    left_avg = average_lane(left_lines, height)
    right_avg = average_lane(right_lines, height)

    lane_center_x = None

    if left_avg and right_avg:
        mid_bottom = ((left_avg[0] + right_avg[0]) // 2, height)
        mid_top = ((left_avg[2] + right_avg[2]) // 2, int(height * 0.6))
        lane_center_x = mid_bottom[0]
    elif left_avg:
        dx = int(width * 0.25)
        lane_center_x = left_avg[0] + dx
    elif right_avg:
        dx = int(width * 0.25)
        lane_center_x = right_avg[0] - dx

    # === 9. Compute steering value ===
    steering_value = 0.0

    if lane_center_x is not None:
        frame_center_x = width // 2
        deviation = lane_center_x - frame_center_x
        steering_value = deviation / (width / 2)  # normalize to [-1, 1]
        steering_value = np.clip(steering_value, -1.0, 1.0)

    return steering_value


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

            car.setSteering(-10 * value)

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
