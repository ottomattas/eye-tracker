import cv2
import numpy as np
import pyautogui
import dlib
from scipy.spatial import distance
import time
from collections import deque

# Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Settings
SMOOTHING_FACTOR = 0.5
CALIBRATION_TIME = 5  # seconds
MOVEMENT_THRESHOLD = 20

class EyeTracker:
    def __init__(self):
        self.calibration_points = []
        self.is_calibrated = False
        self.center_point = None
        self.mouse_positions = deque(maxlen=5)  # Store last 5 positions for smoothing
        
    def get_eye_aspect_ratio(self, eye_points):
        # Calculate the eye aspect ratio
        vertical_dist1 = distance.euclidean(eye_points[1], eye_points[5])
        vertical_dist2 = distance.euclidean(eye_points[2], eye_points[4])
        horizontal_dist = distance.euclidean(eye_points[0], eye_points[3])
        
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    
    def get_eye_coordinates(self, landmarks, frame):
        # Get coordinates for both eyes
        left_eye = []
        right_eye = []
        
        for n in range(36, 42):  # Left eye landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
            
        for n in range(42, 48):  # Right eye landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))
            
        return np.array(left_eye), np.array(right_eye)
    
    def calibrate(self, eye_center):
        if len(self.calibration_points) < 30:  # Collect 30 points for calibration
            self.calibration_points.append(eye_center)
        else:
            # Calculate average center point from calibration
            self.center_point = np.mean(self.calibration_points, axis=0)
            self.is_calibrated = True
            print("Calibration complete!")
    
    def get_mouse_position(self, eye_center):
        if not self.is_calibrated:
            return None
        
        # Calculate offset from center point
        offset_x = eye_center[0] - self.center_point[0]
        offset_y = eye_center[1] - self.center_point[1]
        
        # Scale offset to screen coordinates
        screen_x = screen_width // 2 + (offset_x * 2)
        screen_y = screen_height // 2 + (offset_y * 2)
        
        # Apply smoothing
        self.mouse_positions.append((screen_x, screen_y))
        smoothed_pos = np.mean(self.mouse_positions, axis=0)
        
        return int(smoothed_pos[0]), int(smoothed_pos[1])

def wait_for_camera_access():
    """Wait until camera access is granted."""
    print("Waiting for camera access...")
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("Camera access granted!")
            return True
        else:
            print("Please grant camera access in System Preferences/Security & Privacy/Camera")
            time.sleep(2)  # Wait 2 seconds before trying again

def main():
    # Wait for camera access before proceeding
    wait_for_camera_access()
    
    tracker = EyeTracker()
    # Initialize camera after access is granted
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    print(f"Calibrating for {CALIBRATION_TIME} seconds...")
    print("Please look at the center of your screen")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye, right_eye = tracker.get_eye_coordinates(landmarks, frame)
            
            # Calculate eye centers
            left_eye_center = np.mean(left_eye, axis=0).astype(int)
            right_eye_center = np.mean(right_eye, axis=0).astype(int)
            eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)
            
            # Draw eye landmarks
            for eye in [left_eye, right_eye]:
                cv2.polylines(frame, [eye], True, (0, 255, 0), 1)
            
            # Calibration phase
            if not tracker.is_calibrated:
                if time.time() - start_time < CALIBRATION_TIME:
                    tracker.calibrate(eye_center)
                    cv2.putText(frame, "Calibrating...", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    tracker.is_calibrated = True
            else:
                # Get and apply mouse position
                mouse_pos = tracker.get_mouse_position(eye_center)
                if mouse_pos:
                    x, y = mouse_pos
                    # Ensure coordinates are within screen bounds
                    x = max(0, min(x, screen_width))
                    y = max(0, min(y, screen_height))
                    pyautogui.moveTo(x, y, duration=0.1)
        
        # Display the frame
        cv2.imshow('Eye Tracking', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Fail-safe for PyAutoGUI
    pyautogui.FAILSAFE = True
    main()