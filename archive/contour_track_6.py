import cv2
import time
import numpy as np
from collections import deque
from imutils.video import FPS

class ObjectTracker:
    def __init__(self, video_path, buffer_size=32, barbell_radius_mm=25):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.barbell_radius_mm = barbell_radius_mm
        self.counter = 0
        self.dX, self.dY = 0, 0
        self.direction = ""
        self.line_colour = (0, 0, 255)
        self.pts = deque(maxlen=self.buffer_size)
        self.rep_count = 0
        self.concentric = False
        self.eccentric = False
        self.y_total_distance_rep = []
        self.y_total_velocity_rep = []
        self.y_total_velocity_set = []
        self.y_average_velocity_set = []
        self.y_velocity_estimation = []
        self.rep_time = 0  # Initialize rep_time here

    def initialize_video(self):
        return cv2.VideoCapture(self.video_path)

    def initialize_arguments(self):
        return {'video': self.video_path, 'buffer': self.buffer_size}

    def update_direction(self, frame, mmpp, prev_gray):
        for i in np.arange(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            if self.counter >= 10 and i == 1 and self.pts[-2] is not None:
                self.dX = self.pts[-2][0] - self.pts[i][0]
                self.dY = self.pts[-2][1] - self.pts[i][1]
                dirX, dirY = "", ""
                if np.abs(self.dX) > 20:
                    dirX = "Beginning" if np.sign(self.dX) == 1 else "Ending"
                    self.line_colour = (255, 0, 0) if dirX == "Beginning" else self.line_colour
                if np.abs(self.dY) > 20:
                    dirY = "Concentric" if np.sign(self.dY) == 1 else "Eccentric"
                    self.line_colour = (0, 255, 0) if dirY == "Concentric" else (0, 0, 255)
                self.direction = dirY if dirY else dirX

                thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * 1.25)
                cv2.line(frame, self.pts[i - 1], self.pts[i], self.line_colour, thickness)

        if self.y_velocity > 0.16:
            cv2.putText(frame, f' Y Velocity {self.y_velocity:.2f} m/s)', (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f' NOT MOVING', (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        
        cv2.putText(frame, self.direction, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(self.dX, self.dY), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def main(self):
        cap = self.initialize_video()
        args = self.initialize_arguments()
        frame_count = 0
        start_time = time.time()
        rep_start_time = time.time()
        last_y = None
        video_fps = 30
        moving = False
        started = False
        blue_lower = np.array([70, 20, 20], dtype="uint8")
        blue_upper = np.array([140, 255, 255], dtype="uint8")
        ret, frame = cap.read()
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
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
                (x_circ, y_circ), radius = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if last_y is None:
                    last_y = y_circ
            if radius > 10:
                if last_y != y_circ:
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)
                    cv2.circle(frame, (int(x_circ), int(y_circ)), int(radius), (0, 255, 255), 2)
                ref_radius = radius
                mmpp = self.barbell_radius_mm / ref_radius
                y_disp = last_y - y_circ
                self.y_distance_per_frame = y_disp * mmpp

                if y_disp > self.barbell_radius_mm / 3:
                    moving = True


                if not moving:
                    rep_start_time = time.time()
                self.y_velocity = self.y_distance_per_frame * video_fps / 1000
                self.y_total_distance_rep.append(self.y_distance_per_frame)
                if self.y_velocity > 0.16:
                    moving = True
                    self.y_total_velocity_rep.append(self.y_velocity)
                if moving:
                    if not started:
                        started = True
                        rep_start_time = time.time()
                    self.rep_time = time.time() - rep_start_time
                if self.y_velocity < 0.07 and self.y_velocity > -0.07 and elapsed_time > 0.8 and (
                        sum(self.y_total_distance_rep) > 100 or sum(self.y_total_distance_rep) < -99):
                    cv2.circle(frame, (int(x_circ), int(y_circ)), int(radius), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    self.y_total_velocity_set.append(self.y_velocity)
                    self.y_total_distance_rep = []  # Corrected variable name here
                    start_time = time.time()
                    if moving and self.concentric:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False
                        self.y_total_velocity_set.append(self.y_total_velocity_rep)
                        rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / self.rep_time)
                        self.y_velocity_estimation.append(rep_velocity_estimation)
                        average_velocity = np.average(self.y_total_velocity_rep)
                        self.y_average_velocity_set.append(average_velocity)
                    elif moving and self.eccentric:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False
                       
                        self.y_total_velocity_set.append(self.y_total_velocity_rep)
                        rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / self.rep_time)
                        self.y_velocity_estimation.append(rep_velocity_estimation)
                        average_velocity = np.average(self.y_total_velocity_rep)
                        self.y_average_velocity_set.append(average_velocity)
                      
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
            cv2.putText(frame, f'Elapsed time: {elapsed_time:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Rep Count: {self.rep_count:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Concentric: {self.concentric:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Eccentric: {self.eccentric:.2f}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {moving}', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Moving: {self.rep_time}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.pts.appendleft(center)
            self.update_direction(frame, mmpp, prev_frame)  # Pass prev_frame to update_direction
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            self.counter += 1
            last_y = y_circ
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ObjectTracker(video_path='blue_night_2.webm')
    tracker.main()
