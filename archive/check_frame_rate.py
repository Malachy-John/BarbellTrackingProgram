import cv2
import time
from check_video_length import get_video_length

# Input video path
input_video_path = 'IMG_04_fix.MOV'

# Capture the input video
cap = cv2.VideoCapture(input_video_path)

# Get the original video's frame rate
fps = cap.get(cv2.CAP_PROP_FPS) 

# Calculate the delay between frames in milliseconds and the time per frame in seconds
frame_delay = int(1000 / fps)
time_per_frame = 1.0 / fps

frame_count = 0
start_time = 0


elapsed_seconds = 0

start_time_too = time.time()

total_elapsed_time = 0


while True:
    start_time_frame = time.time()  # Start time of frame processing
    
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no frames left

    # Your object tracking processing here

    frame = cv2.resize(frame, (1280, 720))
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)

    end_time = time.time()  # End time of frame processing
    processing_time = end_time - start_time_frame
    
    # Calculate remaining time to wait to match the desired frame rate
    wait_time = max(frame_delay - int(processing_time * 1000), 1)  # Ensure wait_time is not negative
    
    # Increment frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    

    # Calculate and print FPS every second
    if elapsed_time >= 1.0:  # Every one second
        fps = frame_count / elapsed_time
        print(f"Current FPS: {fps:.2f}")
        frame_count = 0  # Reset frame count
        start_time = time.time()  # Reset start time

    
    total_elapsed_time = time.time() - start_time_too

    if total_elapsed_time >=1.0:
        new_elapsed_seconds = total_elapsed_time
        #print(new_elapsed_seconds)

        if new_elapsed_seconds > elapsed_seconds:
            # A new second has passed
            elapsed_seconds = new_elapsed_seconds
            print(f"Elapsed seconds: {elapsed_seconds}")


    # Wait for a specified time or until a key is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        
        break

# Release everything when the job is finished
cap.release()
cv2.destroyAllWindows()

check_video_length = get_video_length('IMG_04_fix.MOV')

print(f"{int(check_video_length/elapsed_seconds) * 100}%")
