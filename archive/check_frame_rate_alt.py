import cv2
import time

# Assuming get_video_length is appropriately defined in 'check_video_length'
from check_video_length import get_video_length

input_video_path = 'IMG_06_fix.MOV'
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Now frame_delay in nanoseconds (ns)
frame_delay_ns = int(1_000_000_000 / fps)

start_time_ns = time.perf_counter_ns()

elapsed_seconds_ns = 0
frame_count = 0

alt_start_time_ns = time.perf_counter_ns()

while True:
    start_time_frame_ns = time.perf_counter_ns()
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Frame', frame)

    end_time_frame_ns = time.perf_counter_ns()
    processing_time_ns = end_time_frame_ns - start_time_frame_ns

    # wait_time in ns, but cv2.waitKey() needs ms
    wait_time_ms = max((frame_delay_ns - processing_time_ns) // 1_000_000, 1)

    frame_count += 1

    # For FPS calculation and printing, convert to seconds for readability
    if (time.perf_counter_ns() - start_time_ns) >= 1_000_000_000:
        fps = frame_count / ((time.perf_counter_ns() - start_time_ns) / 1_000_000_000)
        print(f"Current FPS: {fps:.2f}")
        frame_count = 0
        start_time_ns = time.perf_counter_ns()

    if cv2.waitKey(int(wait_time_ms)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Assuming get_video_length() returns length in seconds, convert it for comparison or calculation
#video_length_ns = get_video_length(input_video_path) * 1_000_000_000
#total_elapsed_time_ns = time.perf_counter_ns() - start_time_ns


# Calculate total elapsed time in seconds from nanoseconds
total_elapsed_time_ns = time.perf_counter_ns() - alt_start_time_ns
total_elapsed_time_seconds = total_elapsed_time_ns / 1_000_000_000

# Assuming get_video_length returns video length in seconds
video_length_seconds = get_video_length(input_video_path)

print(total_elapsed_time_seconds)

# Final print statement adjusted for the context of your calculation
# (The original calculation might need clarification for its intended purpose)
print(f"{video_length_seconds/total_elapsed_time_seconds}%")
