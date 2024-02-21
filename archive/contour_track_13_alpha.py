import cv2
import numpy as np
import time
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
cap = cv2.VideoCapture('IMG_06_fix.MOV')

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
rep_start_time = time.time()
last_y = None
started = False
frame_rate = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rate += 1

    elapsed_time = time.time() - start_time


    #frame = imutils.resize(frame, width=640, height=640)

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

    # Optionally, draw contours on the original frame
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            radius = min(w, h) // 2
            center = (x + w // 2, y + h // 2)

        

            if last_y is None:
                last_y = center[1]

            if radius > 10:
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
                                rep_start_time = time.time()
                            speeds.append(speed_check * mmpp)
                            
                            

                    elif prev_y is not None and y > prev_y:
                        speed_check = abs(y - prev_y)
                        if speed_check <= 4:
                            end_position = y
                            if started and abs(end_position - start_y_pos) < 90:
                                total = 0
                                started = False
                                print()
                                print(f"YOUR TIME IS: {time.time() - rep_start_time:.2f}")
                                total = np.sum(speeds)
                                total = total / 1000
                                time_value = time.time() - rep_start_time
                                total = total / time_value
                                print(f"Your speed was {total:.2f}")
                                reps += 1
                                #print(f"Velocity of rep: {self.find_value(round(total, 2))}")
                                cv2.putText(frame, f" Speed {total:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                

                                if reps == 1:
                                    start_y_pos = y

                                speeds = []
                                
                            else:
                                continue

            else:
                continue

            if elapsed_time >= 1.0:  # Every second, update the FPS value
                fps = frame_rate / elapsed_time
                print(f"FPS: {fps}")
                frame_rate = 0
                start_time = time.time()

            cv2.putText(frame, f" REPS {reps}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            cv2.imshow('Frame', frame)
            #cv2.imshow('Mask', mask)
            last_y = center[1]
            prev_y = y if y is not None else prev_y



    # Calculate FPS
    
    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time)
    prev_frame_time = curr_frame_time

    # Display FPS on frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



    # Display the result
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
