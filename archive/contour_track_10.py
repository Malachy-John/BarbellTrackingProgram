import cv2
import time
import numpy as np
from collections import deque
import imutils

class ObjectTracker:
    def __init__(self, video_path, buffer_size=32, barbell_radius_mm=50):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.barbell_radius_mm = barbell_radius_mm
        self.pts = deque(maxlen=self.buffer_size)
        self.rep_count = 0
        self.concentric = False
        self.eccentric = False
        self.y_total_distance_rep = []
        self.y_velocity_estimation = []
        self.positive_y = []
        self.rep_time = 0  # Initialize rep_time here

    def find_value(self, input_value):
        if input_value < 0.16:
            return -1
        if input_value == 0.16:
            return 0
        elif 0.16 <= input_value < 0.23:
            return 0.5
        elif input_value == 0.23:
            return 1
        elif 0.23 <= input_value < 0.26:
            return 1.5
        elif input_value == 0.26:
            return 2
        elif 0.26 <= input_value < 0.3:
            return 2.5
        elif input_value == 0.3:
            return 3
        elif 0.3 <= input_value <= 0.34:
            return 3.5
        elif input_value == 0.34:
            return 4
        elif 0.34 <= input_value < 0.38:
            return 4.5
        elif input_value == 0.38:
            return 5
        elif 0.38 <= input_value < 0.42:
            return 5.5
        elif input_value == 0.42:
            return 6
        elif input_value > 0.42:
            return 7
        else:
            return None

    def main(self):
        cap = cv2.VideoCapture(self.video_path)
        rep_start_time = time.time()
        last_y = None
        started = False

        # Define the range of blue color in HSV
        blue_lower = np.array([110, 80, 80])
        blue_upper = np.array([140, 255, 255])
        
        ref_radius = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        x, y, w, h = 0,0,0,0

        prev_y = None
        speeds = []
        avg_speeds = []
        speed_check = 0
        #time = 0

        reps = 0
        start_y_pos = 60

        fps = cap.get(cv2.CAP_PROP_FPS)

        # check for absolute distance moved to verify if its a rep?
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=800)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center = None

            if cnts:
                largest_contour = max(cnts, key=cv2.contourArea)

                # Get the bounding box around the contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Calculate the centroid of the bounding box
                centroid = (int(x + w / 2), int(y + h / 2))

                # Draw the bounding box and centroid on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)


                radius = min(w, h) // 2
                center = (x + w // 2, y + h // 2)

                if last_y is None:
                    last_y = center[1]

                if radius > 10:
                    
                    
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    ref_radius = radius
                    mmpp = self.barbell_radius_mm / ref_radius
                    y_disp = last_y - center[1]
                    self.y_distance_per_frame = y_disp * mmpp
                   
                        
                    if prev_y is not None and y < prev_y:
                            
                            
                            speed_check  = abs(y - prev_y)
                            if speed_check < 8:
                                cv2.putText(frame, f"WE ARE AT A STASIS HERE",(10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                            else:
                                if not started:       
                                    started = True
                                    rep_start_time = time.time()
                                speeds.append(speed_check * mmpp)
                                cv2.putText(frame, f'Moving UP', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                           
                            
                    elif prev_y is not None and y > prev_y:
                        speed_check  = abs(y - prev_y)

                        #if started and speed_check > 3:
                                #print(speed_check)
                            
                        if speed_check <= 4:
                            end_position = y
                            if started and abs(end_position - start_y_pos) < 90:
                                #print(f"end position {y}")
                                started = False
                                print()
                                print(f"YOUR TIME IS: {time.time() - rep_start_time:.2f}")
                                #print(f"Your speeds {np.sum(speeds)}")
                                total = np.sum(speeds)
                                total = total / 1000
                                time_value = time.time() - rep_start_time
                                total = total / time_value
                                print(f"Your speed was {total:.2f}")
                                reps += 1

                                print(f"Velocity of rep: {self.find_value(round(total, 2))}")
                                
                                if reps == 1:
                                    start_y_pos = y

                                speeds = []
                                total = 0
                            else:
                                continue
                                  
            else:
                continue

            cv2.putText(frame, f" REPS {reps}",(10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            self.pts.appendleft(center)
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            last_y = center[1]
            prev_y = y if y is not None else prev_y

            delay = int(1000 / 30)
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
        

        cap.release() 
        cv2.destroyAllWindows()

        print()
        

        for i, v in enumerate(avg_speeds):

            print(f"Velocity of rep {i+1}: {v:.2f}: {self.find_value(round(v, 2))}")

if __name__ == "__main__":
    tracker = ObjectTracker(video_path='big_blue_night_2.webm')
    tracker.main()