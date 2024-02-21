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
        blue_lower = np.array([110, 50, 50])
        blue_upper = np.array([140, 255, 255])
        

        #blue_lower = np.array([30, 50, 50], dtype="uint8")
        #blue_upper = np.array([70, 255, 255], dtype="uint8")
        radius = 0

       
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
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if cnts else []
            center = None
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Draw a perfect circle within the bounding box
                radius = min(w, h) // 2
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)

                if last_y is None:
                    last_y = center[1]

            if radius > 10:
                
                if last_y != center[1]:
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)
                    cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 2)


                
                ref_radius = radius
                mmpp = self.barbell_radius_mm / ref_radius
                y_disp = last_y - center[1]
                self.y_distance_per_frame = y_disp * mmpp

                
                if not moving:
                    rep_start_time = time.time()

                self.y_velocity = self.y_distance_per_frame * video_fps / 1000

                self.y_total_distance_rep.append(self.y_distance_per_frame)

                if  self.y_distance_per_frame > self.barbell_radius_mm / 4:
                    moving = True
                
                

                if moving:

                    if self.y_velocity > 0:
                        self.positive_y.append(self.y_distance_per_frame)
                    
                    if not started:
                        started = True
                        rep_start_time = time.time()
                    self.rep_time = time.time() - rep_start_time

                if self.y_velocity < 0.07 and self.y_velocity > -0.07 and elapsed_time > 0.1 and (
                        sum(self.y_total_distance_rep) > 100 or sum(self.y_total_distance_rep) < -99):
                    cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    
                    start_time = time.time()
                    if moving:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False
                       

                        
                        rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / self.rep_time)
                        
                        
                        
                        self.y_velocity_estimation.append(rep_velocity_estimation)
                        
                       

                    elif moving and self.eccentric:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False

                        


                        rep_velocity_estimation = ((sum(self.positive_y) / 1000) / self.rep_time)
                        

                        self.y_velocity_estimation.append(rep_velocity_estimation)
                        
                        

                elif self.y_velocity > 0.07:
                    self.concentric = True
                    self.eccentric = False
                    if not moving:
                        self.concentric = False
                        self.eccentric = True
                elif self.y_velocity < -0.07:
                    self.eccentric = True
                    self.concentric = False
                    if not moving:
                        self.concentric = False
                        self.eccentric = True
            else:
                continue

            cv2.putText(frame, f'Elapsed time: {elapsed_time:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Rep Count: {self.rep_count:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Concentric: {self.concentric:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Eccentric: {self.eccentric:.2f}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {moving}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {self.rep_time:.3f}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.pts.appendleft(center)
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            last_y = center[1]
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

        print()
        

        for i, v in enumerate(self.y_velocity_estimation):

            print(f"Velocity of rep {i+1}: {v:.2f}: {self.find_value(round(v, 2))}")

if __name__ == "__main__":
    tracker = ObjectTracker(video_path='big_blue_night_6.webm')
    tracker.main()
