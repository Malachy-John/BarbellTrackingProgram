import cv2
import time
import numpy as np
from collections import deque
import argparse
import imutils
import math
from imutils.video import FPS


class ObjectTracker:
    FPS_LIMIT = 15

    def __init__(self, video_path, buffer_size=32, barbell_radius_mm=25):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.barbell_radius_mm = barbell_radius_mm
        self.counter = 0
        self.dX, self.dY = 0, 0
        self.velocity_X, self.velocity_Y = 0, 0
        self.direction = ""
        self.line_colour = (0, 0, 255)
        self.pts = deque(maxlen=self.buffer_size)
        self.pixel_to_meter_conversion = None

    def initialize_video(self):
        cap = cv2.VideoCapture(self.video_path)
        return cap

    def select_reference_frame(self, cap, roi_hist):
        _, frame = cap.read()

        frame = imutils.resize(frame, width=600)
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 255, 255), -1)

        print(radius)
        mmpp = self.barbell_radius_mm / radius

        cv2.imshow('Reference Image', frame)
        cv2.waitKey(0)
        cv2.destroyWindow('Reference Image')

        print(f"reference_radius: {radius:.2f} mmpp: {mmpp:.2f}")
        return radius, mmpp

    def calculate_velocity(self, elapsed_time):
        self.velocity_X = self.dX * 25 / 1000
        self.velocity_Y = self.dY * 25 / 1000

    def select_roi(self, cap):
        _, frame = cap.read()
        roi = cv2.selectROI(frame)
        roi_hsv = cv2.cvtColor(frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        return frame, roi, roi_hist

    def initialize_arguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the (optional) video file")
        ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
        return vars(ap.parse_args())

    def update_direction(self, frame, elapsed_time):
        for i in np.arange(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            if self.counter >= 10 and i == 1 and self.pts[-5] is not None:
                self.dX = self.pts[-5][0] - self.pts[i][0]
                self.dY = self.pts[-5][1] - self.pts[i][1]
                (dirX, dirY) = ("", "")

                if np.abs(self.dX) > 20:
                    dirX = "Beginning" if np.sign(self.dX) == 1 else "Ending"
                    if dirX == "Beginning":
                        self.line_colour = (255, 0, 0)

                if np.abs(self.dY) > 20:
                    dirY = "Concentric" if np.sign(self.dY) == 1 else "Eccentric"
                    if dirY == "Concentric":
                        self.line_colour = (0, 255, 0)
                    elif dirY == "Eccentric":
                        self.line_colour = (0, 0, 255)

                if dirY != "":
                    self.direction = f"{dirY}"
                elif dirY == "" and len(dirX) > 0:
                    self.direction = f"{dirX}"

            thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * 1.25)
            cv2.line(frame, self.pts[i - 1], self.pts[i], self.line_colour, thickness)

        self.calculate_velocity(elapsed_time)

        cv2.putText(frame, f'X Velocity: ({self.velocity_X:.2f} m/s, Y Velocity {self.velocity_Y:.2f} m/s)',
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.putText(frame, self.direction, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(self.dX, self.dY), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def main(self):
        cap = self.initialize_video()
        frame, roi, roi_hist = self.select_roi(cap)
        args = self.initialize_arguments()
        frame_count = 0
        reference_radius, mmpp = self.select_reference_frame(cap, roi_hist)
        start_time = time.time()

        frame_count = 0


        roi_hsv = cv2.cvtColor(frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], cv2.COLOR_BGR2HSV)

        #cv2.imshow(roi_hsv)
        roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        
        
        blue_lower = np.array([80,20,20],dtype="uint8")
        blue_upper = np.array([140,255,255], dtype="uint8")
        

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            #blurred = cv2.GaussianBlur(frame, (11, 11), 0)

            elapsed_time = time.time() - start_time

            #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	        # construct a mask for the color "green", then perform
	        # a series of dilations and erosions to remove any small
	        # blobs left in the mask
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            rect, tracking_window = cv2.CamShift(mask, (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2))
            

            if rect:
                points = cv2.boxPoints(rect)
                points = np.int0(points)


                center = (((points[0][0] + points[2][0])/ 2 ), ((points[0][1] + points[2][1])/2))

                mid = (((points[0][0] + points[1][0])/ 2 ), ((points[0][1] + points[1][1])/2))

                radius = int(math.sqrt((((mid[0] - center[0]) ** 2) + ((mid[1] - center[1]) ** 2))))
                
                print(radius)


                cv2.circle(frame, (int(center[0]), int(center[1])), radius, (0,255,0), 3)
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0,255,0), -1)


                img2 = cv2.polylines(frame, [points], True, (0, 255, 0), 2)

                self.pts.appendleft(points[0])
                self.update_direction(frame, elapsed_time)

                cv2.imshow('Frame', img2)
                cv2.imshow("mask", mask)
                self.counter += 1
            else:
                rect, tracking_window = cv2.CamShift(mask, (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

            if cv2.waitKey(30) & 0xFF == 27:
                break



        cap.release()
        cv2.destroyAllWindows()

        print()
        expected_speeds = ("Very Slow", "Slow", "Medium", "Fast", "Very Fast")

        for i, v in enumerate(self.y_velocity_estimation):

            if i == 0 or i == 9:
                choice = 1
            elif i == 6:
                choice = 0
            elif i in (2,3,4,5,8):
                choice = 4
            else:
                choice = 2

            print(f"Velocity of rep {i+1}: {v:.2f}: {expected_speeds[choice]}")


if __name__ == "__main__":
    tracker = ObjectTracker(video_path='blue_card.webm')
    tracker.main()
