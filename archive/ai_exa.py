import cv2
import numpy as np
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change 0 to the camera index if you have multiple cameras

# Define the blue color range in HSV
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Barbell width in millimeters
barbell_width_mm = 50.0

# Initialize variables for speed calculation
repetition_start_time = time.time()
prev_centroid = None
total_distance_pixels = 0.0
total_time_elapsed = 0.0
rep_count = 0
average_speeds = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to extract the blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour (assuming it's the barbell)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box around the contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the centroid of the bounding box
        centroid = (int(x + w / 2), int(y + h / 2))

        # Draw the bounding box and centroid on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

        # Calculate speed in meters per second
        if prev_centroid:
            distance_pixels = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            total_distance_pixels += distance_pixels
            total_time_elapsed = time.time() - repetition_start_time

        prev_centroid = centroid

    # Display the frame
    cv2.imshow("Barbell Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check for the end of a repetition (for example, based on a certain time threshold)
    if total_time_elapsed > 5.0:  # Change this threshold based on your specific scenario
        # Calculate average speed for the current repetition
        if total_time_elapsed > 0:
            rep_count += 1
            total_distance_meters = (total_distance_pixels / w) * barbell_width_mm / 1000.0
            average_speed = total_distance_meters / total_time_elapsed
            average_speeds.append(average_speed)

        # Reset variables for the next repetition
        repetition_start_time = time.time()
        total_distance_pixels = 0.0
        total_time_elapsed = 0.0
        prev_centroid = None

# Display average speeds at the end
if rep_count > 0:
    average_speed_mps = sum(average_speeds) / rep_count
    print(f"Average Speed Across {rep_count} Repetitions: {average_speed_mps:.2f} m/s")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
