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
            return 11
        if input_value == 0.16:
            return 10
        elif 0.16 <= input_value < 0.23:
            return 9.5
        elif input_value == 0.23:
            return 9
        elif 0.23 <= input_value < 0.26:
            return 8.5
        elif input_value == 0.26:
            return 8
        elif 0.26 <= input_value < 0.3:
            return 7.5
        elif input_value == 0.3:
            return 7
        elif 0.3 <= input_value <= 0.34:
            return 6.5
        elif input_value == 0.34:
            return 6
        elif 0.34 <= input_value < 0.38:
            return 5.5
        elif input_value == 0.38:
            return 5
        elif 0.38 <= input_value < 0.42:
            return 4.5
        elif input_value == 0.42:
            return 4
        elif input_value > 0.42:
            return 3
        else:
            return None

    def main(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        start_time = time.time()
        rep_start_time = time.time()
        last_y = None
        video_fps = 30
        moving = False
        started = False
        # Define the range of blue color in HSV
        blue_lower = np.array([110, 90, 90])
        blue_upper = np.array([150, 255, 255])
        

        #blue_lower = np.array([30, 50, 50], dtype="uint8")
        #blue_upper = np.array([70, 255, 255], dtype="uint8")
        ref_radius = None
        #radius = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        x, y, w, h = 0,0,0,0

        prev_y = None
        speeds = []
        avg_speeds = []
        fps = 30

        # check for absolute distance moved to verify if its a rep?
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame = imutils.resize(frame, width=800)

            elapsed_time = time.time() - start_time
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            #mask = cv2.erode(mask, None, iterations=2)
            #mask = cv2.dilate(mask, None, iterations=2)


            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]

            
            #cnts = cnts[0] if cnts else []

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

                    
                    if not moving:
                        
                        rep_start_time = time.time()

                    self.y_velocity = self.y_distance_per_frame * video_fps / 1000

                    self.y_total_distance_rep.append(self.y_distance_per_frame)

                    if  self.y_distance_per_frame > self.barbell_radius_mm / 3:
                        moving = True
                        if not started:
                            started = True
                            rep_start_time = time.time()
                    
                    
                    if moving:
                        self.rep_time = time.time() - rep_start_time

                        if prev_y is not None and y < prev_y:
                            speed = prev_y - y
                            speed_mm = speed * mmpp
                            speeds.append(speed_mm)

                        if self.y_velocity < 0.05 and self.y_velocity > -0.05:
                            print(sum(self.y_total_distance_rep))
                            final_rep_time = self.rep_time
                            rep_start_time = time.time()

                            
                            self.rep_count += 1
                            self.concentric = False
                            self.eccentric = True
                            moving = False
                            started = False
                            
                            #avg_speed = np.mean(speeds) if speeds else 0

                            total_speed = np.sum(speeds) if speeds else 0

                            total_speed = total_speed / 1000

                            avg_speed = total_speed / final_rep_time

                            avg_speeds.append(avg_speed)
                                
                            rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / final_rep_time)
                                
                                
                                
                            self.y_velocity_estimation.append(rep_velocity_estimation)
            else:
                continue

            
            cv2.putText(frame, f'Rep Count: {self.rep_count:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Eccentric: {self.eccentric:.2f}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {moving}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {self.rep_time:.3f}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.pts.appendleft(center)
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            last_y = center[1]
            prev_y = y if y is not None else prev_y
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        print()
        

        for i, v in enumerate(avg_speeds):

            print(f"Velocity of rep {i+1}: {v:.2f}: {self.find_value(round(v, 2))}")

if __name__ == "__main__":
    tracker = ObjectTracker(video_path='big_blue_night_6.webm')
    tracker.main()
