import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import urllib.request
import os

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Camera settings
cam_w, cam_h = 640, 480
frame_reduction = 100  # Reduce frame area for better control

# Smoothing parameters
smoothing = 7
prev_x, prev_y = 0, 0

# Click detection
click_threshold = 30  # Distance threshold for click gesture
clicking = False

# Download hand_landmarker.task if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model...")
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded!")

# Initialize MediaPipe HandLandmarker
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process frame with MediaPipe
    results = detector.detect(mp_image)
    
    # Draw landmarks and control mouse
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Get index finger tip (landmark 8) and thumb tip (landmark 4)
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            
            # Convert normalized coordinates to pixel coordinates
            index_x = int(index_tip.x * cam_w)
            index_y = int(index_tip.y * cam_h)
            
            # Map camera coordinates to screen coordinates
            # Only use the center area of the frame for better control
            screen_x = np.interp(index_x, [frame_reduction, cam_w - frame_reduction], [0, screen_w])
            screen_y = np.interp(index_y, [frame_reduction, cam_h - frame_reduction], [0, screen_h])
            
            # Apply smoothing
            curr_x = prev_x + (screen_x - prev_x) / smoothing
            curr_y = prev_y + (screen_y - prev_y) / smoothing
            
            # Move mouse
            pyautogui.moveTo(curr_x, curr_y)
            
            # Update previous position
            prev_x, prev_y = curr_x, curr_y
            
            # Click detection: measure distance between index finger and thumb
            thumb_x = int(thumb_tip.x * cam_w)
            thumb_y = int(thumb_tip.y * cam_h)
            
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Draw circle on index finger tip and thumb
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
            
            # Draw all hand landmarks
            for landmark in hand_landmarks:
                lx = int(landmark.x * cam_w)
                ly = int(landmark.y * cam_h)
                cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)
            
            # Click when fingers are close together
            if distance < click_threshold:
                if not clicking:
                    pyautogui.click()
                    clicking = True
                    cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), 3)
            else:
                clicking = False
    
    # Display FPS
    cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('Hand Tracking Mouse Control', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector = None
