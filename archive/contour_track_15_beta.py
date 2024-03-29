import cv2
import numpy as np
import time
from starting_pos import find_initial_coordinates

def upload_to_gpu(frame):
    """Upload frame to GPU."""
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    return gpu_frame

def download_from_gpu(gpu_frame):
    """Download frame from GPU to CPU."""
    return gpu_frame.download()

def convert_color_gpu(gpu_frame, conversion_code):
    """Convert frame color space on GPU."""
    return cv2.cuda.cvtColor(gpu_frame, conversion_code)

def is_inside_bounding_box(x, y, start_x, start_y, start_w, start_h):
    """
    Check if the point (x, y) is inside the bounding box defined by
    the top-left corner (start_x, start_y) with width start_w and height start_h.

    Parameters:
    x (int): X coordinate of the point.
    y (int): Y coordinate of the point.
    start_x (int): X coordinate of the top-left corner of the bounding box.
    start_y (int): Y coordinate of the top-left corner of the bounding box.
    start_w (int): Width of the bounding box.
    start_h (int): Height of the bounding box.

    Returns:
    bool: True if the point is inside the bounding box, False otherwise.
    """
    if x >= start_x and x < start_x + start_w and y >= start_y and y < start_y + start_h:
        return True
    else:
        return False


# Start video capture
cap = cv2.VideoCapture('IMG_05_fix.MOV')

start_x, start_y, start_w, start_h = find_initial_coordinates('IMG_04_fix.MOV')

print(f"x = {start_x}, y = {start_y}, w = {start_w}, h = {start_h}")

# Define the range of blue color in HSV
lower_blue = np.array([75,90,90])
upper_blue = np.array([110,255,255])

prev_frame_time = time.time()
curr_frame_time = time.time()

prev_y = None
speeds = []
avg_speeds = []
speed_check = 0
total = 0
reps = 0
start_y_pos = 60
barbell_radius_mm = 50
rep_start_time_ns = time.perf_counter_ns()  # Adjusted for high-precision timing
last_y = []
started = False
frame_rate = 0
start_time_ns = time.perf_counter_ns()  # Adjusted for high-precision timing

fps = cap.get(cv2.CAP_PROP_FPS)

# Now frame_delay in nanoseconds (ns)
frame_delay_ns = int(1_000_000_000 / fps)

while True:
    start_time_frame_ns = time.perf_counter_ns()
    ret, frame = cap.read()

    
    if not ret:
        break

    frame_rate += 1

    # Upload frame to GPU
    gpu_frame = upload_to_gpu(frame)

    # Convert BGR to HSV using GPU
    hsv_gpu = convert_color_gpu(gpu_frame, cv2.COLOR_BGR2HSV)

    # Download the HSV frame back to CPU for thresholding
    hsv = download_from_gpu(hsv_gpu)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours (on CPU) to track the blue object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centroid = (int(start_x + start_w / 2), int(start_y + start_h / 2))
    
    cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if start_x == x and start_y == y:
                print("STARTING ZONE")
            centroid = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            radius = min(start_w, start_h) // 2
            center = (x + w // 2, y + h // 2)

            # this needs a little bit of work
            if is_inside_bounding_box(centroid[0],centroid[1], start_x, start_y, start_w, start_h):
                #print("Inside the starting position")
                cv2.rectangle(frame, (start_x, start_y), (start_x+start_w, start_y+start_h), (255, 0, 255), 2)

            if last_y is None:
                last_y.append(center[1])

            
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            ref_radius = radius
            mmpp = barbell_radius_mm / ref_radius
            y_disp = last_y - center[1]
            y_distance_per_frame = y_disp * mmpp

            if prev_y is not None and y < prev_y:
                speed_check = abs(y - prev_y)
                if speed_check < 8:
                    cv2.putText(frame, f"", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                else:
                    if not started:
                        started = True
                        rep_start_time_ns = time.perf_counter_ns()
                    speeds.append(speed_check * mmpp)
            elif prev_y is not None and y > prev_y:
                speed_check = abs(y - prev_y)
                if speed_check <= 4:
                    end_position = y
                    if started and abs(end_position - start_y_pos) < 90:
                        total = 0
                        started = False
                        rep_duration_ns = time.perf_counter_ns() - rep_start_time_ns  # Rep duration in nanoseconds
                        print(f"YOUR TIME IS: {rep_duration_ns / 1_000_000_000:.2f} seconds")  # Convert ns to seconds
                        total = np.sum(speeds)
                        total = total / 1000
                        time_value = rep_duration_ns / 1_000_000_000  # Convert ns to seconds for time value
                        total = total / time_value
                        print(f"Your speed was {total:.2f} m/s")
                        reps += 1
                        print(f"Reps: {reps}")
                        cv2.putText(frame, f" Speed {total:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if reps == 1:
                            start_y_pos = y
                        speeds = []
                    else:
                        continue
            

            last_y = center[1]
            prev_y = y if y is not None else prev_y

    end_time_frame_ns = time.perf_counter_ns()
    processing_time_ns = end_time_frame_ns - start_time_frame_ns

    # wait_time in ns, but cv2.waitKey() needs ms
    wait_time_ms = max((frame_delay_ns - processing_time_ns) // 1_000_000, 1)

    #frame_count += 1

    # Display the result
    cv2.imshow('Frame', frame)

    if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Note: FPS calculation and elapsed time reporting were removed to focus on keeping all your original processing logic intact.
