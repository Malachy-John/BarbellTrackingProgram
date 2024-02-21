import cv2
import time
import numpy as np
from collections import deque
import argparse
import imutils
from imutils.video import FPS

class ObjectTracker:
    """Class for tracking a moving object in a video."""

    def __init__(self, video_path, buffer_size=32, barbell_radius_mm=25):
        """Initialize the ObjectTracker class.

        Parameters:
        - video_path (str): Path to the video file.
        - buffer_size (int): Maximum size of the buffer for tracking points.
        - barbell_radius_mm (int): Radius of the barbell in millimeters.
        """

        # Video-related attributes
        self.video_path = video_path  # Path to the video file
        self.buffer_size = buffer_size  # Maximum size of the buffer for tracking points
        self.barbell_radius_mm = barbell_radius_mm  # Radius of the barbell in millimeters

        # Tracking and velocity attributes
        self.counter = 0  # Counter for tracking frames
        self.dX, self.dY = 0, 0  # Displacement in X and Y directions
        self.x_velocity, self.y_velocity = 0, 0  # Velocity in X and Y directions
        self.direction = ""  # Current motion direction
        self.line_colour = (0, 0, 255)  # Color for drawing tracking lines
        self.pts = deque(maxlen=self.buffer_size)  # Buffer for storing tracking points

        # Repetition counting and motion state attributes
        self.rep_count = 0  # Count of repetitions
        self.concentric = False  # Flag for concentric motion
        self.eccentric = False  # Flag for eccentric motion

        # Lists for storing motion metrics
        self.y_total_distance = []  # Total Y-axis distance over all frames
        self.y_total_distance_rep = []  # Y-axis distance per repetition
        self.y_total_velocity_rep = []  # Y-axis velocity per repetition
        self.y_total_velocity_set = []  # Y-axis velocity set
        self.y_average_velocity_set = []  # Average Y-axis velocity set

        self.y_distance_per_frame = 0  # Y-axis distance per frame
        self.rep_time = 0  # Time taken for a single repetition

        # List for storing velocity estimations
        self.y_velocity_estimation = []  # Y-axis velocity estimations


    def initialize_video(self):
        """Initialize video capture using OpenCV.

        Returns:
        - cv2.VideoCapture: Video capture object.
        """
        cap = cv2.VideoCapture(self.video_path)
        return cap

    def select_reference_frame(self, cap, roi_hist):
        """Select a reference frame for motion analysis.

        Parameters:
        - cap (cv2.VideoCapture): Video capture object.
        - roi_hist (numpy.ndarray): Histogram of the Region of Interest (ROI) in HSV color space.

        Returns:
        - Tuple[float, float]: Reference radius and millimeters per pixel (mmpp).
        """
        _, frame = cap.read()
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

        mmpp = self.barbell_radius_mm / radius

        cv2.imshow('Reference Image', frame)
        cv2.waitKey(0)
        cv2.destroyWindow('Reference Image')

        print(f"reference_radius: {radius:.2f} mmpp: {mmpp:.2f}")
        return radius, mmpp

    def select_roi(self, cap):
        """Select a Region of Interest (ROI) for tracking.

        Parameters:
        - cap (cv2.VideoCapture): Video capture object.

        Returns:
        - Tuple[tuple, numpy.ndarray]: Selected ROI coordinates and its histogram in HSV color space.
        """
        _, frame = cap.read()
        roi = cv2.selectROI(frame)
        roi_hsv = cv2.cvtColor(frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
        return roi, roi_hist

    def initialize_arguments(self):
        """Parse command line arguments.

        Returns:
        - dict: Parsed command line arguments.
        """
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the (optional) video file")
        ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
        return vars(ap.parse_args())

    def update_direction(self, frame, mmpp):
        """Update and display motion direction on the video frame.

        Parameters:
        - frame (numpy.ndarray): Input video frame.
        - mmpp (float): Millimeters per pixel (mmpp).

        Returns:
        - None
        """

        # Iterate through the tracked points
        for i in np.arange(1, len(self.pts)):
            # Check for valid points
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            # Analyze motion direction when sufficient points are available
            if self.counter >= 10 and i == 1 and self.pts[-2] is not None:
                self.dX = self.pts[-2][0] - self.pts[i][0]
                self.dY = self.pts[-2][1] - self.pts[i][1]
                (dirX, dirY) = ("", "")

                # Determine motion direction in X-axis
                if np.abs(self.dX) > 20:
                    dirX = "Beginning" if np.sign(self.dX) == 1 else "Ending"
                    if dirX == "Beginning":
                        self.line_colour = (255, 0, 0)  # Set line color for beginning motion

                # Determine motion direction in Y-axis
                if np.abs(self.dY) > 20:
                    dirY = "Concentric" if np.sign(self.dY) == 1 else "Eccentric"
                    if dirY == "Concentric":
                        self.line_colour = (0, 255, 0)  # Set line color for concentric motion
                    elif dirY == "Eccentric":
                        self.line_colour = (0, 0, 255)  # Set line color for eccentric motion

                # Update the motion direction attribute
                if dirY != "":
                    self.direction = f"{dirY}"
                elif dirY == "" and len(dirX) > 0:
                    self.direction = f"{dirX}"

            # Calculate and draw tracking lines with varying thickness
            thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * 1.25)
            cv2.line(frame, self.pts[i - 1], self.pts[i], self.line_colour, thickness)

        # Display Y-axis velocity information
        if self.y_velocity > 0.16:
            cv2.putText(frame, f' Y Velocity {self.y_velocity:.2f} m/s)',
                        (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f' NOT MOVING',
                        (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)

        # Display motion direction and displacement information
        cv2.putText(frame, self.direction, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(self.dX, self.dY), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


    def main(self):
        """Main function for running the object tracking algorithm."""
        
        cap = self.initialize_video()
        roi, roi_hist = self.select_roi(cap)
        args = self.initialize_arguments()
        frame_count = 0
        reference_radius, mmpp = self.select_reference_frame(cap, roi_hist)
        start_time = time.time()

        rep_start_time = time.time()
        last_y = None
        video_fps = 30
        moving = False
        started = False

        blue_lower = np.array([80,80,80],dtype="uint8")
        blue_upper = np.array([140,255,255], dtype="uint8")


        #blue_lower = np.array([40,20,20],dtype="uint8")
        #blue_upper = np.array([65,255,255], dtype="uint8")

        
        # Infinite loop to process each frame of the video
        while True:
            # Read the current frame from the video capture
            ret, frame = cap.read()
            # cv2.waitKey(delay_time)

            # Break the loop if the frame is not successfully captured
            if not ret:
                break

            # Preprocess the frame for motion analysis
            frame_count += 1
            #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            elapsed_time = time.time() - start_time
            #hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Generate a mask based on the region of interest (ROI) histogram
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	        # construct a mask for the color "green", then perform
	        # a series of dilations and erosions to remove any small
	        # blobs left in the mask
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours in the mask to identify the object's position
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            # Process the contours if any are found
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x_circ, y_circ), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # Initialize the last_y variable if not already set
                if last_y is None:
                    last_y = y_circ

            # Process the detected object if its radius is above a certain threshold
            if radius > 10:
                # Draw circles and calculate relevant metrics
                if last_y != y_circ:
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)
                    cv2.circle(frame, (int(x_circ), int(y_circ)), int(radius), (0, 255, 255), 2)

                ref_radius = radius
                mmpp = self.barbell_radius_mm / ref_radius
                y_disp = last_y - y_circ
                self.y_distance_per_frame = y_disp * mmpp

                # Initialize or update the repetition start time
                if not moving:
                    rep_start_time = time.time()

                # Check for unusual distance and calculate velocity
                if self.y_distance_per_frame > 100:
                    print("WHAT THE HECK")

                self.y_velocity = self.y_distance_per_frame * video_fps / 1000

                # Update cumulative distance and velocity lists
                self.y_total_distance.append(self.y_distance_per_frame)
                self.y_total_distance_rep.append(self.y_distance_per_frame)

                if self.y_velocity > 0.16:
                    moving = True
                    self.y_total_velocity_rep.append(self.y_velocity)

                # Update repetition time if currently moving
                if moving:
                    if started == False:
                        started = True
                        rep_start_time = time.time()
                    self.rep_time = time.time() - rep_start_time

                # Check for the end of a repetition and update relevant metrics
                if self.y_velocity < 0.07 and self.y_velocity > -0.07 and elapsed_time > 0.8 and (
                        sum(self.y_total_distance) > 100 or sum(self.y_total_distance) < -99):
                    cv2.circle(frame, (int(x_circ), int(y_circ)), int(radius), (255, 0, 0), 2)
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    print(f"average_distance: {np.average(self.y_total_distance):.2f}")

                    print(self.y_total_velocity_rep)

                    # Update velocity and distance sets, and reset lists
                    self.y_total_velocity_set.append(self.y_velocity)
                    self.y_total_distance = []
                    start_time = time.time()

                    # Check for concentric motion and update repetition count
                    if moving == True and self.concentric:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False
                        print(f"REPETITION TIME TAKEN {self.rep_time}")
                        print(f"average_velocity: {np.average(self.y_total_velocity_rep):.2f}")
                        self.y_total_velocity_set.append(self.y_total_velocity_rep)

                        # Estimate repetition velocity and update lists
                        rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / self.rep_time)
                        self.y_velocity_estimation.append(rep_velocity_estimation)
                        average_velocity = np.average(self.y_total_velocity_rep)
                        self.y_average_velocity_set.append(average_velocity)
                        print(f'Your total distance: {sum(self.y_total_distance)}')

                    elif moving == True and self.eccentric:
                        self.rep_count += 1
                        self.concentric = False
                        self.eccentric = True
                        moving = False
                        started = False
                        print(f"REPETITION TIME TAKEN {self.rep_time}")
                        print(f"average_velocity: {np.average(self.y_total_velocity_rep):.2f}")
                        self.y_total_velocity_set.append(self.y_total_velocity_rep)

                        rep_velocity_estimation = ((sum(self.y_total_distance_rep) / 1000) / self.rep_time)

                        self.y_velocity_estimation.append(rep_velocity_estimation)

                        average_velocity = np.average(self.y_total_velocity_rep)

                        self.y_average_velocity_set.append(average_velocity)
                        print(f'Your total distance: {sum(self.y_total_distance)}')

                # Check for concentric and eccentric motion
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

            # Display various information on the video frame
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

            # Update and display motion direction on the video frame
            self.update_direction(frame, mmpp)

            # Display the video frames and mask
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            self.counter += 1

            last_y = y_circ
            # Break the loop if the 'Esc' key is pressed
            if cv2.waitKey(30) & 0xFF == 27:
                break

        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

        


if __name__ == "__main__":
    tracker = ObjectTracker(video_path='final_day_blue_2.webm')
    tracker.main()
