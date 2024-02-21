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
    """
    if x >= start_x and x < start_x + start_w and y >= start_y and y < start_y + start_h:
        return True
    else:
        return False

# Start video capture
cap = cv2.VideoCapture('IMG_11_fix.MOV')

start_x, start_y, start_w, start_h = find_initial_coordinates('IMG_11_fix.MOV')
print(f"x = {start_x}, y = {start_y}, w = {start_w}, h = {start_h}")

# Define the range of blue color in HSV
lower_blue = np.array([75, 90, 90])
upper_blue = np.array([100, 255, 255])

prev_y = None
speeds = []
avg_speeds = []
speed_check = 0
total = 0
reps = 0
start_y_pos = 60
barbell_radius_mm = 50
rep_start_time_ns = time.perf_counter_ns()

y_positions = deque(maxlen=2000)
started = False
frame_rate = 0
start_time_ns = time.perf_counter_ns()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay_ns = int(1_000_000_000 / fps)
frame_width = 1280

set_started = False
concentric_started = False

y_pos_list = []

rep_count = 0
eccentric_started = False
end_x, end_y, finish_x, finish_y = None, None, None, None

drawn_line = False

frame_count = 0

prev_x = None
x_positions = []
print(fps)

while True:
    start_time_frame_ns = time.perf_counter_ns()
    ret, frame = cap.read()

    if not ret:
        break


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #gpu_frame = upload_to_gpu(frame)
    #hsv_gpu = convert_color_gpu(gpu_frame, cv2.COLOR_BGR2HSV)
    #hsv = download_from_gpu(hsv_gpu)
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centroid = (int(start_x + start_w / 2), int(start_y + start_h / 2))
    cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            if is_inside_bounding_box(center[0], center[1], start_x, start_y, start_w, start_h):
                cv2.rectangle(frame, (start_x, start_y), (start_x+start_w, start_y+start_h), (255, 0, 255), 2)

            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            x_positions.append(center[0])
            y_positions.append(center[1])

            if prev_y is None:
                prev_y = y_positions[0]
            if prev_x is None:
                prev_x = x_positions[0]
            
            if len(x_positions) > 1:
                prev_x = x_positions[len(x_positions) - 2]

            if(len(y_positions) > 1):
                prev_y = y_positions[len(y_positions) - 2]

            if is_inside_bounding_box(center[0], center[1], start_x, start_y, start_w, start_h):
                cv2.rectangle(frame, (start_x, start_y), (start_x+start_w, start_y+start_h), (255, 0, 255), 2)

            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            tracked_y = center[1]
            ref_radius = min(start_w, start_h) // 2
            end_point = (center[0] + ref_radius, center[1])
            cv2.line(frame, center, end_point, (0, 255, 0), 2)
            mmpp = barbell_radius_mm / ref_radius
            y_disp = prev_y - center[1]
            x_disp = prev_x - center[0]
            y_distance_per_frame = y_disp * mmpp

            if not set_started:
                if y_disp < -2 and x_disp < -2:
                    print("Rep started")
                    rep_ending_y_pos = min(y_positions) + (ref_radius)
                    set_started = True

            if set_started:
                start_point = (0, rep_ending_y_pos)
                end_point = (frame_width - 1, rep_ending_y_pos)
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

            if set_started and not concentric_started:
                if y_disp < -2:
                    eccentric_started = True
                    distance_check = center[1] - rep_ending_y_pos
                    if distance_check < -60:
                        rep_ending_y_pos = center[1] + int(ref_radius)           
                if y_distance_per_frame > 4 and eccentric_started:
                    end_x, end_y = center[0], center[1]
                    concentric_started = True
                    rep_start_time_ns = time.perf_counter_ns()

            elif set_started and concentric_started:
                frame_count += 1
                if tracked_y <= rep_ending_y_pos:
                    print("end of rep")
                    print(f"FRAME COUNT: {frame_count}")
                    concentric_started = False
                    eccentric_started = False          
                    rep_count += 1
                    rep_duration_ns = time.perf_counter_ns() - rep_start_time_ns
                    print(f"YOUR TIME IS: {rep_duration_ns / 1_000_000_000:.2f} seconds")
                    rep_duration_s = rep_duration_ns / 1_000_000_000

                    actual_fps = frame_count / rep_duration_s

                    print(f"Your actual fps is {actual_fps}")

                    print(f"PERCENTAGE: {(fps/actual_fps)}")

                    adjustment_percentage = (fps/actual_fps)

                    finish_x, finish_y = center[0], center[1]
                    distance_metres = (abs(end_y - finish_y) * mmpp) / 1000
                    metres_per_second = (distance_metres / rep_duration_s) * adjustment_percentage
                    print(f"Your m/s value is: {metres_per_second:.3f}")
                    frame_count = 0

            cv2.putText(frame, f"Repetitions: {rep_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rep_start_path = (end_x, end_y)
            rep_end_path = (finish_x, finish_y)
            if finish_x is not None and finish_y is not None:
                cv2.line(frame, rep_start_path, rep_end_path, (255, 255, 0), 2)

    end_time_frame_ns = time.perf_counter_ns()
    processing_time_ns = end_time_frame_ns - start_time_frame_ns
    wait_time_ms = max((frame_delay_ns - processing_time_ns) // 1_000_000, 1)
    #frame_count += 1
    cv2.imshow('Frame', frame)
    if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
