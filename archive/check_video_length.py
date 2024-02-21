import cv2


def get_video_length(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
    else:
        # Get total number of video frames
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Get the video's FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the video's total duration in seconds
        duration_seconds = total_frames / fps
        
        print(f"Total Duration: {duration_seconds:.2f} seconds")

    # Release the video capture object
    cap.release()
    return duration_seconds