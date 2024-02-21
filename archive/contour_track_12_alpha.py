import cv2
import numpy as np
import imutils

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

# Start video capture
cap = cv2.VideoCapture("IMG_01.MOV")

# Define the range of blue color in HSV
lower_blue = np.array([90,90,90])
upper_blue = np.array([110,255,255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640, height=640)

    # Upload frame to GPU
    gpu_frame = upload_to_gpu(frame)

    # Convert BGR to HSV using GPU
    hsv_gpu = convert_color_gpu(gpu_frame, cv2.COLOR_BGR2HSV)

    # Download the HSV frame back to CPU for thresholding
    hsv = download_from_gpu(hsv_gpu)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # You can perform morphological operations here if needed, using CPU or GPU.
    # For GPU, you would use cv2.cuda.createMorphologyFilter then apply it.

    # Find contours (on CPU) to track the blue object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Optionally, draw contours on the original frame
    for contour in contours:
        # Calculate contour area and ignore small areas
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Frame', frame)
    # cv2.imshow('Mask', mask)  # If you want to see the mask

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
