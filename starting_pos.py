import cv2
import numpy as np

def find_initial_coordinates(video_path):
    """
    Finds the starting x, y coordinates of an object in the first frame of a video.
    
    Args:
    - video_path: Path to the video file.
    - lower_color: Lower bound of the color range in HSV.
    - upper_color: Upper bound of the color range in HSV.
    
    Returns:
    A tuple (x, y) of the initial coordinates, or None if no object is found.
    """

    lower_blue = np.array([75, 90, 90])
    upper_blue = np.array([110, 255, 255])
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    if not success:
        print("Failed to read video")
        cap.release()
        return None

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cap.release()

    # If contours are found, return the coordinates of the first detected object
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)
    
    return None

# Example usage


