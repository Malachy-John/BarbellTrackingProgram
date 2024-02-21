import cv2
import numpy as np
import time
from starting_pos import find_initial_coordinates
from collections import deque

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.start_x, self.start_y, self.start_w, self.start_h = find_initial_coordinates(video_path)
        self.lower_blue = np.array([75, 90, 90])
        self.upper_blue = np.array([100, 255, 255])
        self.prev_y = None
        self.prev_x = None
        self.y_positions = deque(maxlen=2000)
        self.x_positions = []
        self.frame_count = 0
        self.rep_count = 0
        self.set_started = False
        self.concentric_started = False
        self.eccentric_started = False
        self.barbell_radius_mm = 50
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay_ns = int(1_000_000_000 / self.fps)
        self.frame_width = 1280
        self.rep_start_time_ns = None
        self.bottom_x = None
        self.bottom_y = None
        self.top_finish_x = None
        self.top_finish_y = None
        self.rep_ending_y_pos = None
        self.metres_per_second_list = []

    def is_inside_bounding_box(self, x, y):
        return self.start_x <= x < self.start_x + self.start_w and self.start_y <= y < self.start_y + self.start_h

    def process_frame(self, frame):
        start_time_frame_ns = time.perf_counter_ns()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centroid = (int(self.start_x + self.start_w / 2), int(self.start_y + self.start_h / 2))
        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                center = (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                if self.is_inside_bounding_box(center[0], center[1]):
                    cv2.rectangle(frame, (self.start_x, self.start_y), (self.start_x+self.start_w, self.start_y+self.start_h), (255, 0, 255), 2)

                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                self.x_positions.append(center[0])
                self.y_positions.append(center[1])

                if self.prev_y is None:
                    self.prev_y = self.y_positions[0]
                if self.prev_x is None:
                    self.prev_x = self.x_positions[0]

                if len(self.x_positions) > 1:
                    self.prev_x = self.x_positions[len(self.x_positions) - 2]

                if len(self.y_positions) > 1:
                    self.prev_y = self.y_positions[len(self.y_positions) - 2]

                if self.is_inside_bounding_box(center[0], center[1]):
                    cv2.rectangle(frame, (self.start_x, self.start_y), (self.start_x+self.start_w, self.start_y+self.start_h), (255, 0, 255), 2)

                tracked_y = center[1]
                ref_radius = min(self.start_w, self.start_h) // 2
                end_point = (center[0] + ref_radius, center[1])
                cv2.line(frame, center, end_point, (0, 255, 0), 2)
                mmpp = self.barbell_radius_mm / ref_radius
                y_disp = self.prev_y - center[1]
                x_disp = self.prev_x - center[0]
                y_distance_per_frame = y_disp * mmpp

                if not self.set_started:
                    if y_disp < -2 and x_disp < -2:
                        print("Rep started")
                        self.rep_ending_y_pos = min(self.y_positions) + (ref_radius)
                        self.set_started = True

                if self.set_started:
                    start_point = (0, self.rep_ending_y_pos)
                    end_point = (self.frame_width - 1, self.rep_ending_y_pos)
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

                if self.set_started and not self.concentric_started:
                    if y_disp < -2:
                        self.eccentric_started = True
                        distance_check = center[1] - self.rep_ending_y_pos
                        #if distance_check < -60:
                        #    self.rep_ending_y_pos = center[1] + int(ref_radius)           
                    if y_distance_per_frame > 4 and self.eccentric_started:
                        self.bottom_x, self.bottom_y = center[0], center[1]
                        self.concentric_started = True
                        self.rep_start_time_ns = time.perf_counter_ns()

                elif self.set_started and self.concentric_started:
                    self.frame_count += 1
                    if tracked_y <= self.rep_ending_y_pos:
                        #print("end of rep")
                        #print(f"FRAME COUNT: {self.frame_count}")
                        self.concentric_started = False
                        self.eccentric_started = False          
                        self.rep_count += 1
                        rep_duration_ns = time.perf_counter_ns() - self.rep_start_time_ns
                        #print(f"YOUR TIME IS: {rep_duration_ns / 1_000_000_000:.2f} seconds")
                        rep_duration_s = rep_duration_ns / 1_000_000_000

                        actual_fps = self.frame_count / rep_duration_s

                        #print(f"Your actual fps is {actual_fps}")

                        #print(f"PERCENTAGE: {(self.fps/actual_fps)}")

                        adjustment_percentage = (self.fps/actual_fps)

                        self.top_finish_x, self.top_finish_y = center[0], center[1]
                        distance_metres = (abs(self.bottom_y - self.top_finish_y) * mmpp) / 1000
                        metres_per_second = (distance_metres / rep_duration_s) * adjustment_percentage
                        print(f"Your m/s value is: {metres_per_second:.3f}")

                        self.metres_per_second_list.append(metres_per_second)
                        self.frame_count = 0

                cv2.putText(frame, f"Repetitions: {self.rep_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                rep_start_path = (self.bottom_x, self.bottom_y)
                rep_end_path = (self.top_finish_x, self.top_finish_y)
                if self.top_finish_x is not None and self.top_finish_y is not None:
                    cv2.line(frame, rep_start_path, rep_end_path, (255, 255, 0), 2)

                if self.is_inside_bounding_box(x, y) and self.set_started:
                    #print("Set ended")
                    continue

        end_time_frame_ns = time.perf_counter_ns()
        processing_time_ns = end_time_frame_ns - start_time_frame_ns
        wait_time_ms = max((self.frame_delay_ns - processing_time_ns) // 1_000_000, 1)
        cv2.imshow('Frame', frame)

        return wait_time_ms

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            wait_time_ms = self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

        return self.metres_per_second_list

if __name__ == "__main__":
    video_path = 'IMG_11_fix.MOV'
    processor = VideoProcessor(video_path)
    mps_list = processor.run()

    print(mps_list)
