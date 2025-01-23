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
        self.is_paused = False
        self.sensitivity = 2.0  # Default sensitivity multiplier
        self.min_sensitivity = 0.5
        self.max_sensitivity = 4.0
        self.at_sensitivity_limit = False
        self.limit_warning_time = 0
        self.calibration_progress = 0  # Track calibration progress
        
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
        if not eye_center.any():  # Check if eye_center is valid
            return False
            
        if len(self.calibration_points) < 30:  # Collect 30 points for calibration
            self.calibration_points.append(eye_center)
            self.calibration_progress = (len(self.calibration_points) / 30) * 100
            return False
        else:
            # Calculate average center point from calibration
            self.center_point = np.mean(self.calibration_points, axis=0)
            self.is_calibrated = True
            print("Calibration complete!")
            return True
    
    def get_mouse_position(self, eye_center):
        if not self.is_calibrated or self.is_paused:
            return None
        
        try:
            # Calculate offset from center point
            offset_x = eye_center[0] - self.center_point[0]
            offset_y = eye_center[1] - self.center_point[1]
            
            # Scale offset to screen coordinates using sensitivity
            screen_x = screen_width // 2 + (offset_x * self.sensitivity)
            screen_y = screen_height // 2 + (offset_y * self.sensitivity)
            
            # Apply smoothing
            self.mouse_positions.append((screen_x, screen_y))
            smoothed_pos = np.mean(self.mouse_positions, axis=0)
            
            return int(smoothed_pos[0]), int(smoothed_pos[1])
        except (ValueError, TypeError, OverflowError) as e:
            print(f"Error calculating mouse position: {e}")
            return None

    def adjust_sensitivity(self, increase=True):
        old_sensitivity = self.sensitivity
        if increase:
            self.sensitivity = min(self.max_sensitivity, self.sensitivity + 0.2)
        else:
            self.sensitivity = max(self.min_sensitivity, self.sensitivity - 0.2)
        
        # Check if we hit a limit
        if self.sensitivity in [self.min_sensitivity, self.max_sensitivity]:
            self.at_sensitivity_limit = True
            self.limit_warning_time = time.time()
        else:
            self.at_sensitivity_limit = False

def draw_ui(frame, tracker, frame_height):
    # Calculate dimensions based on frame size
    frame_width = frame.shape[1]
    ui_width = min(500, int(frame_width * 0.9))  # Increased max width
    ui_height = 120  # Reduced height
    padding = 15
    text_height = 30  # Increased text height for better separation
    
    # Calculate positions for two-column layout
    ui_x = padding
    ui_y = padding
    left_col_x = ui_x + padding
    right_col_x = ui_x + ui_width - 200  # Fixed position for right column
    
    # Draw semi-transparent black background for UI
    ui_overlay = frame.copy()
    cv2.rectangle(ui_overlay, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (0, 0, 0), -1)
    cv2.addWeighted(ui_overlay, 0.5, frame, 0.5, 0, frame)  # Increased opacity
    
    # Left Column: Status and Info
    # Draw status
    status = "PAUSED" if tracker.is_paused else ("CALIBRATING" if not tracker.is_calibrated else "TRACKING")
    cv2.putText(frame, f"Status: {status}", 
                (left_col_x, ui_y + text_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw calibration progress or sensitivity (left column, second line)
    if not tracker.is_calibrated:
        # Progress text
        progress_text = f"Calibration: {tracker.calibration_progress:.0f}%"
        cv2.putText(frame, progress_text, 
                    (left_col_x, ui_y + text_height * 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Progress bar (shortened and moved up)
        bar_y = ui_y + text_height * 2.5
        bar_length = 150  # Fixed shorter length
        filled_length = int(bar_length * tracker.calibration_progress / 100)
        
        # Draw background bar
        cv2.rectangle(frame, 
                     (left_col_x, int(bar_y)), 
                     (left_col_x + bar_length, int(bar_y + 8)), 
                     (100, 100, 100), 1)
        
        # Draw filled portion
        if filled_length > 0:
            cv2.rectangle(frame, 
                         (left_col_x, int(bar_y)), 
                         (left_col_x + filled_length, int(bar_y + 8)), 
                         (0, 255, 0), -1)
    else:
        # Sensitivity text
        sensitivity_text = f"Sensitivity: {tracker.sensitivity:.1f}"
        cv2.putText(frame, sensitivity_text, 
                    (left_col_x, ui_y + text_height * 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Range in smaller text below
        range_text = f"Range: [{tracker.min_sensitivity:.1f}-{tracker.max_sensitivity:.1f}]"
        cv2.putText(frame, range_text, 
                    (left_col_x, ui_y + text_height * 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Right Column: Controls
    controls = [
        "Q: Quit",
        "R: Recalibrate",
        "P: Pause/Resume",
        "+/-: Sensitivity"
    ]
    for i, control in enumerate(controls):
        cv2.putText(frame, control, 
                    (right_col_x, ui_y + text_height * (i + 1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show warning if at sensitivity limit (as overlay)
    if tracker.at_sensitivity_limit and time.time() - tracker.limit_warning_time < 1.0:
        limit_msg = "Maximum sensitivity" if tracker.sensitivity == tracker.max_sensitivity else "Minimum sensitivity"
        # Draw warning with background
        warning_y = ui_y + ui_height + padding
        cv2.putText(frame, limit_msg, 
                    (left_col_x, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

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
        
        if len(faces) > 0:  # Only process if a face is detected
            face = faces[0]  # Use the first face detected
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
                else:
                    tracker.is_calibrated = True
            elif not tracker.is_paused:  # Only move mouse if not paused and calibrated
                # Get and apply mouse position
                mouse_pos = tracker.get_mouse_position(eye_center)
                if mouse_pos:
                    x, y = mouse_pos
                    # Ensure coordinates are within screen bounds
                    x = max(0, min(x, screen_width))
                    y = max(0, min(y, screen_height))
                    pyautogui.moveTo(x, y, duration=0.1)
        
        # Draw UI
        draw_ui(frame, tracker, frame.shape[0])
        
        # Display the frame
        cv2.imshow('Eye Tracking', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.calibration_points = []
            tracker.is_calibrated = False
            tracker.calibration_progress = 0
            start_time = time.time()
            print("Recalibrating...")
        elif key == ord('p'):
            tracker.is_paused = not tracker.is_paused
            print("Tracking Paused" if tracker.is_paused else "Tracking Resumed")
        elif key == ord('=') or key == ord('+'):
            tracker.adjust_sensitivity(increase=True)
        elif key == ord('-') or key == ord('_'):
            tracker.adjust_sensitivity(increase=False)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Fail-safe for PyAutoGUI
    pyautogui.FAILSAFE = True
    main()