import cv2

# Input and output video paths
input_video_path = 'IMG_07.MOV'
output_video_path = 'IMG_07_fix.MOV'

# Desired resolution
new_width = 1280
new_height = 720

# Capture the input video
cap = cv2.VideoCapture(input_video_path)

# Get the original video's frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original FPS: {fps}")  # Verify the detected frame rate

# Optional: Manually set fps if automatic detection is incorrect
# fps = 30  # Uncomment and adjust this line as needed

# Define the codec and create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop if there are no frames left

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Write the resized frame to the output video
    out.write(resized_frame)

# Release everything when the job is finished
cap.release()
out.release()
