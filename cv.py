import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully. Close the window to quit.")

# Create figure for matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Hand Detection")

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            
            # Get hand center (wrist position)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, c = frame.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            
            # Draw circle at wrist
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            
            # Add text
            cv2.putText(frame, "Hand Detected", (cx - 50, cy - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert BGR to RGB for matplotlib
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def update_frame(i):
    frame = process_frame()
    if frame is not None:
        ax.clear()
        ax.imshow(frame)
        ax.set_title("Hand Detection")
        ax.axis('off')

# Create animation
ani = FuncAnimation(fig, update_frame, interval=50)
plt.tight_layout()
plt.show()

# Release resources
cap.release()
print("Camera closed. Program ended.")