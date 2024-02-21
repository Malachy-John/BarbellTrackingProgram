import cv2
import numpy as np
import time
from starting_pos import find_initial_coordinates
from collections import deque

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
cap = cv2.VideoCapture('IMG_04_fix.MOV')

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


y_positions = deque(maxlen=2000)
started = False
frame_rate = 0
start_time_ns = time.perf_counter_ns()  # Adjusted for high-precision timing

fps = cap.get(cv2.CAP_PROP_FPS)

# Now frame_delay in nanoseconds (ns)
frame_delay_ns = int(1_000_000_000 / fps)

frame_width = 1280

rep_started = False

bottom_pos_found = False
bottom_not_found = True

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



    # starting position circle
    centroid = (int(start_x + start_w / 2), int(start_y + start_h / 2))
    
    cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            radius = min(start_w, start_h) // 2
            #center = (x + w // 2, y + h // 2)

            y_positions.append(center[1])

            if prev_y is None:
                prev_y = y_positions[0]

            if(len(y_positions) > 1):
                prev_y = y_positions[len(y_positions) - 2]
                #print(prev_y)

            # this needs a little bit of work
            if is_inside_bounding_box(center[0],center[1], start_x, start_y, start_w, start_h):
                #print("Inside the starting position")
                cv2.rectangle(frame, (start_x, start_y), (start_x+start_w, start_y+start_h), (255, 0, 255), 2)

            
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            tracked_y = center[1]

            

            ref_radius = radius
            mmpp = barbell_radius_mm / ref_radius
            y_disp = prev_y - center[1]
            y_distance_per_frame = y_disp * mmpp

            print(f"Dispacement: {y_disp}")

            if not rep_started:
                if y_disp < -1:
                    print("Rep started")
                    print(y_positions)
                    rep_ending_y_pos = min(y_positions) - (ref_radius//2)
                    rep_started = True

            if rep_started:
                start_point = (0, rep_ending_y_pos)  # Starting at the left edge of the image
                end_point = (frame_width - 1, rep_ending_y_pos)  # Ending at the right edge of the image
                cv2.line(frame, start_point, end_point, (255,0,0), 2)

            if rep_started:
                if y_disp > 2:
                    print("Rep started")
                    print(y_positions)
                    bottom_rep_ending_y_pos = max(y_positions) - (ref_radius//2)
                    bottom_pos_found = True

            if bottom_pos_found and bottom_not_found:
                bottom_start_point = (0, bottom_rep_ending_y_pos)  # Starting at the left edge of the image
                bottom_end_point = (frame_width - 1, bottom_rep_ending_y_pos)  # Ending at the right edge of the image
                cv2.line(frame, bottom_start_point, bottom_end_point, (255,0,0), 2)
                bottom_not_found = False


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
            
            prev_y = y if y is not None else prev_y

    end_time_frame_ns = time.perf_counter_ns()
    processing_time_ns = end_time_frame_ns - start_time_frame_ns

    # wait_time in ns, but cv2.waitKey() needs ms
    wait_time_ms = max((frame_delay_ns - processing_time_ns) // 1_000_000, 1)

    #frame_count += 1

    # Display the result
    cv2.imshow('Frame', frame)

    if cv2.waitKey(0) & 0xFF == 32:  # Space bar ASCII value
        continue  # Go to the next iteration of the loop, thus the next frame

cap.release()
cv2.destroyAllWindows()

# Note: FPS calculation and elapsed time reporting were removed to focus on keeping all your original processing logic intact.
