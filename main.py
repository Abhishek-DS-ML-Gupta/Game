import pygame, sys, random, os, wave, math, struct, time
from pygame import gfxdraw
import numpy as np
# ===== Optional CV2/Mediapipe imports (graceful fallback) =====
USE_HANDS = True
try:
    import cv2
    import mediapipe as mp
    # Suppress TensorFlow/MediaPipe warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress absl warnings
except Exception as e:
    print(f"Error importing OpenCV/MediaPipe: {e}")
    USE_HANDS = False
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
W, H = 800, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Car Dodge - Hand Control (Index Finger)")
clock = pygame.time.Clock()
os.makedirs("sounds", exist_ok=True)
os.makedirs("fonts", exist_ok=True)
os.makedirs("cars", exist_ok=True)

# Load custom fonts
def load_font(path, size):
    try:
        return pygame.font.Font(path, size)
    except:
        return pygame.font.SysFont("arial", size)

# Load the provided fonts
gomarice_font = load_font("fonts/gomarice_game_music_love.ttf", 36)
deadcrt_font = load_font("fonts/DEADCRT.ttf", 48)
future_font = load_font("fonts/Future TimeSplitters.otf", 32)
sharpshooter_font = load_font("fonts/Sharpshooter.otf", 28)
dynatecha_font = load_font("fonts/Dynatecha-Regular.ttf", 14)

# Small fonts for UI
small_gomarice = load_font("fonts/gomarice_game_music_love.ttf", 20)
small_deadcrt = load_font("fonts/DEADCRT.ttf", 24)
small_future = load_font("fonts/Future TimeSplitters.otf", 20)
small_sharpshooter = load_font("fonts/Sharpshooter.otf", 20)
small_dynatecha = load_font("fonts/Dynatecha-Regular.ttf", 10)

# Game states
STATE_CAR_SELECTION = 0
STATE_NAME_INPUT = 1
STATE_GAME = 2
STATE_GAME_OVER = 3
STATE_VICTORY = 4
STATE_END_STORY = 5  # New state for the end story

# Colors for dark theme
BACKGROUND_COLOR = (15, 15, 25)
PANEL_COLOR = (30, 30, 45)
ACCENT_COLOR = (100, 100, 255)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (255, 255, 100)
WARNING_COLOR = (255, 100, 100)
STORY_BACKGROUND = (20, 20, 40)
STORY_TEXT_COLOR = (255, 255, 255)

# Level lighting conditions (light, medium, dark)
LEVEL_LIGHTING = [
    {"name": "light", "sky": (135, 206, 235), "horizon": (255, 255, 200), "ambient": 1.0},  # Day
    {"name": "medium", "sky": (70, 80, 120), "horizon": (255, 180, 100), "ambient": 0.7},  # Sunset
    {"name": "dark", "sky": (10, 10, 30), "horizon": (30, 30, 60), "ambient": 0.4}  # Night
]

# ---------- Camera detection function ----------
def check_camera_available():
    """Check if a camera is available and accessible"""
    try:
        import cv2
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        
        # Check if camera was opened successfully
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        return ret
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return False
    except Exception as e:
        print(f"Camera error: {e}")
        return False

# ---------- helpers to auto-make assets if missing ----------
def make_img(path, color, size=(60, 90), is_player=False, car_type=0):
    surf = pygame.Surface(size, pygame.SRCALPHA)
    
    # Car body with gradient
    for i in range(size[1]):
        gradient_factor = 1 - (i / size[1]) * 0.3
        gradient_color = tuple(int(c * gradient_factor) for c in color)
        pygame.draw.line(surf, gradient_color, (0, i), (size[0], i))
    
    # Car details based on type
    if is_player:
        # Player car details
        pygame.draw.rect(surf, (255,255,255,180), (size[0]//6, 10, size[0]*2//3, 10), border_radius=4)
        pygame.draw.rect(surf, (0,0,0,200), (5, size[1]-14, size[0]-10, 9), border_radius=4)
        
        # Different car designs based on type
        if car_type == 0:  # Sports car
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//4, size[1]//3),
                (size[0]//4, size[1]//2),
                (size[0]*3//4, size[1]//2),
                (size[0]*3//4, size[1]//3)
            ])
            # Spoiler
            pygame.draw.rect(surf, (50, 50, 50), (size[0]//4, 5, size[0]//2, 5))
            
        elif car_type == 1:  # Sedan
            # Windshield
            pygame.draw.polygon(surf, (80, 120, 200, 200), [
                (size[0]//3, size[1]//3),
                (size[0]//3, size[1]//2),
                (size[0]*2//3, size[1]//2),
                (size[0]*2//3, size[1]//3)
            ])
            # Roof
            pygame.draw.rect(surf, (70, 110, 190, 200), (size[0]//3, size[1]//4, size[0]//3, size[1]//4))
            
        elif car_type == 2:  # SUV
            # Windshield
            pygame.draw.polygon(surf, (90, 130, 210, 200), [
                (size[0]//4, size[1]//4),
                (size[0]//4, size[1]//2),
                (size[0]*3//4, size[1]//2),
                (size[0]*3//4, size[1]//4)
            ])
            # Roof rack
            pygame.draw.rect(surf, (40, 40, 40), (size[0]//4, size[1]//5, size[0]//2, 3))
            
        elif car_type == 3:  # Convertible
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//3, size[1]//3),
                (size[0]//3, size[1]//2),
                (size[0]*2//3, size[1]//2),
                (size[0]*2//3, size[1]//3)
            ])
            # No roof - just seats
            pygame.draw.ellipse(surf, (180, 50, 50), (size[0]//3, size[1]//3, size[0]//3, size[1]//6))
            
        elif car_type == 4:  # Truck
            # Cabin
            pygame.draw.rect(surf, (70, 110, 190, 200), (size[0]//4, size[1]//4, size[0]//2, size[1]//4))
            # Bed
            pygame.draw.rect(surf, (60, 60, 60), (size[0]//4, size[1]//2, size[0]//2, size[1]//4))
            
        elif car_type == 5:  # Van
            # Large windshield
            pygame.draw.polygon(surf, (80, 120, 200, 200), [
                (size[0]//5, size[1]//4),
                (size[0]//5, size[1]//2),
                (size[0]*4//5, size[1]//2),
                (size[0]*4//5, size[1]//4)
            ])
            # Sliding door
            pygame.draw.rect(surf, (60, 100, 180), (size[0]//3, size[1]//3, size[0]//3, size[1]//6))
            
        elif car_type == 6:  # Luxury car
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//4, size[1]//3),
                (size[0]//4, size[1]//2),
                (size[0]*3//4, size[1]//2),
                (size[0]*3//4, size[1]//3)
            ])
            # Chrome details
            pygame.draw.rect(surf, (200, 200, 200), (size[0]//6, 15, size[0]*2//3, 3))
            pygame.draw.rect(surf, (200, 200, 200), (size[0]//6, size[1]-15, size[0]*2//3, 3))
            
        elif car_type == 7:  # Racing car
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//4, size[1]//3),
                (size[0]//4, size[1]//2),
                (size[0]*3//4, size[1]//2),
                (size[0]*3//4, size[1]//3)
            ])
            # Racing stripes
            pygame.draw.rect(surf, (255, 0, 0), (size[0]//2-2, 5, 4, size[1]-10))
            # Spoiler
            pygame.draw.polygon(surf, (50, 50, 50), [
                (size[0]//4, 5),
                (size[0]//4, 15),
                (size[0]//3, 15),
                (size[0]//3, 5)
            ])
            pygame.draw.polygon(surf, (50, 50, 50), [
                (size[0]*2//3, 5),
                (size[0]*2//3, 15),
                (size[0]*3//4, 15),
                (size[0]*3//4, 5)
            ])
            
        elif car_type == 8:  # Classic car
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//3, size[1]//3),
                (size[0]//3, size[1]//2),
                (size[0]*2//3, size[1]//2),
                (size[0]*2//3, size[1]//3)
            ])
            # Classic details
            pygame.draw.circle(surf, (200, 200, 200), (size[0]//4, size[1]//2), 5)
            pygame.draw.circle(surf, (200, 200, 200), (size[0]*3//4, size[1]//2), 5)
            
        elif car_type == 9:  # Futuristic car
            # Windshield
            pygame.draw.polygon(surf, (100, 150, 255, 200), [
                (size[0]//4, size[1]//3),
                (size[0]//4, size[1]//2),
                (size[0]*3//4, size[1]//2),
                (size[0]*3//4, size[1]//3)
            ])
            # Neon lights
            pygame.draw.rect(surf, (0, 255, 255), (0, size[1]-5, size[0], 2))
            pygame.draw.rect(surf, (0, 255, 255), (0, 5, size[0], 2))
            # Glow effect
            for i in range(3):
                alpha = 100 - i * 30
                s = pygame.Surface((size[0]+10, size[1]+10), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 255, 255, alpha), (0, 0, size[0]+10, size[1]+10), border_radius=10)
                surf.blit(s, (-5, -5))
    else:
        # Enemy car details
        pygame.draw.rect(surf, (50,50,50,180), (size[0]//6, 10, size[0]*2//3, 8), border_radius=4)
        pygame.draw.rect(surf, (200,200,200,200), (5, size[1]-12, size[0]-10, 7), border_radius=4)
    
    pygame.image.save(surf, path)

def make_wav(path, freq, duration=0.2):
    sr = 44100
    frames = int(sr * duration)
    w = wave.open(path, 'w')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    for i in range(frames):
        val = int(30000 * math.sin(2 * math.pi * freq * i / sr))
        w.writeframesraw(struct.pack('<h', val))
    w.close()

def make_engine_sound(path, base_freq, car_type, duration=1.0):
    """Create a unique engine sound for each car type"""
    sr = 44100
    frames = int(sr * duration)
    w = wave.open(path, 'w')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    
    # Create unique engine sound based on car type
    for i in range(frames):
        t = i / sr
        
        # Base engine frequency varies with car type
        if car_type == 0:  # Sports car - high rev
            freq = base_freq * (1.0 + 0.5 * math.sin(20 * t))
        elif car_type == 1:  # Sedan - smooth
            freq = base_freq * (1.0 + 0.1 * math.sin(5 * t))
        elif car_type == 2:  # SUV - low rumble
            freq = base_freq * 0.8 * (1.0 + 0.2 * math.sin(3 * t))
        elif car_type == 3:  # Convertible - medium rev
            freq = base_freq * (1.0 + 0.3 * math.sin(10 * t))
        elif car_type == 4:  # Truck - diesel sound
            freq = base_freq * 0.7 * (1.0 + 0.4 * math.sin(2 * t))
        elif car_type == 5:  # Van - steady
            freq = base_freq * 0.9 * (1.0 + 0.1 * math.sin(4 * t))
        elif car_type == 6:  # Luxury - smooth high
            freq = base_freq * 1.1 * (1.0 + 0.15 * math.sin(8 * t))
        elif car_type == 7:  # Racing - very high rev
            freq = base_freq * 1.3 * (1.0 + 0.6 * math.sin(25 * t))
        elif car_type == 8:  # Classic - old engine
            freq = base_freq * 0.85 * (1.0 + 0.25 * math.sin(7 * t))
        elif car_type == 9:  # Futuristic - electric whine
            freq = base_freq * 1.2 * (1.0 + 0.4 * math.sin(15 * t))
        
        # Add harmonics for realistic engine sound
        val = int(20000 * (
            math.sin(2 * math.pi * freq * t) + 
            0.5 * math.sin(4 * math.pi * freq * t) + 
            0.3 * math.sin(6 * math.pi * freq * t) +
            0.1 * math.sin(8 * math.pi * freq * t)
        ))
        
        w.writeframesraw(struct.pack('<h', val))
    w.close()

# Auto-generate minimal assets if not present
# Generate 10 different player cars
car_colors = [
    (0, 140, 255),    # Blue sports car
    (255, 0, 0),      # Red sedan
    (0, 100, 0),      # Green SUV
    (255, 215, 0),    # Yellow convertible
    (150, 75, 0),     # Brown truck
    (128, 128, 128),  # Gray van
    (75, 0, 130),     # Purple luxury car
    (255, 140, 0),    # Orange racing car
    (165, 42, 42),    # Brown classic car
    (0, 255, 255)     # Cyan futuristic car
]
car_names = [
    "Sports Car",
    "Sedan",
    "SUV",
    "Convertible",
    "Truck",
    "Van",
    "Luxury Car",
    "Racing Car",
    "Classic Car",
    "Futuristic Car"
]
car_stats = [
    {"speed": 9, "handling": 8, "description": "Fast and agile"},
    {"speed": 6, "handling": 7, "description": "Balanced performance"},
    {"speed": 5, "handling": 5, "description": "Sturdy and reliable"},
    {"speed": 7, "handling": 6, "description": "Open-top freedom"},
    {"speed": 4, "handling": 4, "description": "Heavy but powerful"},
    {"speed": 3, "handling": 3, "description": "Spacious and practical"},
    {"speed": 7, "handling": 9, "description": "Premium comfort"},
    {"speed": 10, "handling": 7, "description": "Built for the track"},
    {"speed": 5, "handling": 6, "description": "Timeless design"},
    {"speed": 8, "handling": 8, "description": "Next-gen technology"}
]

# Generate player cars
for i in range(10):
    car_path = f"cars/car_{i}.png"
    if not os.path.exists(car_path):
        make_img(car_path, car_colors[i], is_player=True, car_type=i)
    
    # Generate unique engine sound for each car
    engine_sound_path = f"sounds/engine_{i}.wav"
    if not os.path.exists(engine_sound_path):
        base_freq = 100 + i * 20  # Different base frequency for each car
        make_engine_sound(engine_sound_path, base_freq, i)

# Generate enemy cars
for i, name in enumerate(["cars/e.png","cars/e1.png","cars/e2.png","cars/e3.png","cars/e4.png"]):
    if not os.path.exists(name):
        colors = [(255,60,60), (60,200,60), (255,200,60), (180,60,200), (60,120,255)]
        make_img(name, colors[i%len(colors)], size=(50, 70))
        
# Generate sound effects
if not os.path.exists("sounds/beep.wav"):
    make_wav("sounds/beep.wav", 800)
    
if not os.path.exists("sounds/crash.wav"):
    make_wav("sounds/crash.wav", 200, 0.5)
if not os.path.exists("sounds/background.wav"):
    # Create a simple background sound
    make_wav("sounds/background.wav", 220, 1.0)
if not os.path.exists("sounds/select.wav"):
    make_wav("sounds/select.wav", 600, 0.1)
if not os.path.exists("sounds/music.wav"):
    # Create background music
    make_wav("sounds/music.wav", 440, 3.0)

# ---------- load images & sounds ----------
def load_scaled(path, size):
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, size)

# Load player cars
PLAYER_CARS = []
for i in range(10):
    car_path = f"cars/car_{i}.png"
    PLAYER_CARS.append(load_scaled(car_path, (100, 90)))

ENEMY_IMGS = [load_scaled(p, (80, 80)) for p in ["cars/e.png","cars/e1.png","cars/e2.png","cars/e3.png","cars/e4.png"]]

def load_sound_chain(candidates):
    for p in candidates:
        try:
            if os.path.exists(p):
                return pygame.mixer.Sound(p)
        except Exception:
            continue
    return None

snd_beep = load_sound_chain(["sounds/beep.wav"]) or pygame.mixer.Sound("sounds/beep.wav")
snd_crash = load_sound_chain(["sounds/crash.wav"]) or pygame.mixer.Sound("sounds/crash.wav")
snd_background = load_sound_chain(["sounds/background.wav"])
snd_select = load_sound_chain(["sounds/select.wav"]) or pygame.mixer.Sound("sounds/select.wav")

# Load engine sounds for each car
ENGINE_SOUNDS = []
for i in range(10):
    engine_path = f"sounds/engine_{i}.wav"
    ENGINE_SOUNDS.append(load_sound_chain([engine_path]))

# ---------- Hand tracker (index fingertip x -> 0..1) ----------
class HandIndexTracker:
    def __init__(self):
        self.enabled = USE_HANDS
        self.last_x_norm = 0.5
        self.smooth_x = 0.5
        self.calibrated = False
        self.calibration_values = []
        
        # Improved pinch detection variables
        self.pinch_detected = False
        self.last_pinch_time = 0
        self.pinch_cooldown = 1.5  # seconds between allowed pinches
        self.pinch_threshold = 0.05  # distance threshold for pinch
        self.pinch_confidence = 0  # for smoother detection
        self.pinch_history = []  # store recent pinch states for stability
        self.pinch_history_length = 5  # number of frames to consider
        
        # Closed fist detection variables
        self.fist_detected = False
        self.last_fist_time = 0
        self.fist_cooldown = 1.5  # seconds between allowed fists
        self.fist_history = []  # store recent fist states for stability
        self.fist_history_length = 5  # number of frames to consider
        
        if not self.enabled:
            print("Hand tracking is disabled")
            return
            
        try:
            # Try different camera indices
            self.cap = None
            for camera_index in range(0, 3):  # Try camera indices 0, 1, 2
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap is not None and self.cap.isOpened():
                    print(f"Using camera at index {camera_index}")
                    break
                else:
                    if self.cap is not None:
                        self.cap.release()
            
            if self.cap is None or not self.cap.isOpened():
                print("Error: Could not open any camera")
                self.enabled = False
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.draw = mp.solutions.drawing_utils
            
            print("Hand tracking initialized successfully")
        except Exception as e:
            print(f"Error initializing hand tracking: {e}")
            self.enabled = False
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        
    def calibrate(self):
        """Calibrate hand position based on initial readings"""
        if not self.enabled:
            return
            
        self.calibration_values = []
        for _ in range(10):
            x_norm = self._get_raw_x_norm()
            if x_norm is not None:
                self.calibration_values.append(x_norm)
            time.sleep(0.1)
            
        if self.calibration_values:
            avg = sum(self.calibration_values) / len(self.calibration_values)
            self.smooth_x = avg
            self.calibrated = True
            
    def _get_raw_x_norm(self):
        """Get raw x position without smoothing"""
        if not self.enabled:
            return None
            
        try:
            ok, frame = self.cap.read()
            if not ok:
                return None
                
            frame = cv2.flip(frame, 1)  # mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                idx_tip = lm.landmark[8]  # Index finger tip
                thumb_tip = lm.landmark[4]  # Thumb tip
                middle_tip = lm.landmark[12]  # Middle finger tip
                
                x = max(0.0, min(1.0, idx_tip.x))
                
                # Check for pinch gesture (thumb and middle finger close together)
                thumb_middle_dist = math.sqrt(
                    (thumb_tip.x - middle_tip.x)**2 + 
                    (thumb_tip.y - middle_tip.y)**2
                )
                
                # Improved pinch detection with confidence and history
                current_time = time.time()
                
                # Update pinch history
                is_pinching = thumb_middle_dist < self.pinch_threshold
                self.pinch_history.append(is_pinching)
                if len(self.pinch_history) > self.pinch_history_length:
                    self.pinch_history.pop(0)
                
                # Check if pinch is stable (most recent frames agree)
                stable_pinch = sum(self.pinch_history) >= self.pinch_history_length * 0.8
                
                # Detect pinch gesture for restart
                if stable_pinch and (current_time - self.last_pinch_time) > self.pinch_cooldown:
                    self.pinch_detected = True
                    self.last_pinch_time = current_time
                    self.pinch_history = []  # Reset history after detection
                    return "restart"  # Special return value for restart
                else:
                    self.pinch_detected = False
                
                # Check for closed fist gesture (all fingers folded)
                # Finger tips: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
                # Finger middle joints: 6 (index), 10 (middle), 14 (ring), 18 (pinky)
                fingers_folded = 0
                if lm.landmark[8].y > lm.landmark[6].y:  # Index finger
                    fingers_folded += 1
                if lm.landmark[12].y > lm.landmark[10].y:  # Middle finger
                    fingers_folded += 1
                if lm.landmark[16].y > lm.landmark[14].y:  # Ring finger
                    fingers_folded += 1
                if lm.landmark[20].y > lm.landmark[18].y:  # Pinky
                    fingers_folded += 1
                
                # Update fist history
                is_fist = fingers_folded >= 3  # At least 3 fingers folded
                self.fist_history.append(is_fist)
                if len(self.fist_history) > self.fist_history_length:
                    self.fist_history.pop(0)
                
                # Check if fist is stable (most recent frames agree)
                stable_fist = sum(self.fist_history) >= self.fist_history_length * 0.8
                
                # Detect fist gesture for exit
                if stable_fist and (current_time - self.last_fist_time) > self.fist_cooldown:
                    self.fist_detected = True
                    self.last_fist_time = current_time
                    self.fist_history = []  # Reset history after detection
                    return "exit"  # Special return value for exit
                else:
                    self.fist_detected = False
                
                # Draw hand landmarks for debugging
                self.draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (int(x*frame.shape[1]), int(idx_tip.y*frame.shape[0])), 8, (255,255,255), -1)
                
                # Show calibration status
                status = "Calibrated" if self.calibrated else "Calibrating..."
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show pinch status with confidence
                pinch_status = f"PINCH: {int(stable_pinch * 100)}%"
                pinch_color = (0, 255, 0) if stable_pinch else (0, 200, 200)
                cv2.putText(frame, pinch_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pinch_color, 2)
                
                # Show fist status with confidence
                fist_status = f"FIST: {int(stable_fist * 100)}%"
                fist_color = (255, 0, 0) if stable_fist else (200, 0, 0)
                cv2.putText(frame, fist_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fist_color, 2)
                
                # Draw circles on thumb and middle finger when pinching
                if stable_pinch:
                    cv2.circle(frame, (int(thumb_tip.x*frame.shape[1]), int(thumb_tip.y*frame.shape[0])), 10, (0, 255, 0), 2)
                    cv2.circle(frame, (int(middle_tip.x*frame.shape[1]), int(middle_tip.y*frame.shape[0])), 10, (0, 255, 0), 2)
                    cv2.line(frame, 
                            (int(thumb_tip.x*frame.shape[1]), int(thumb_tip.y*frame.shape[0])),
                            (int(middle_tip.x*frame.shape[1]), int(middle_tip.y*frame.shape[0])),
                            (0, 255, 0), 2)
                
                # Draw circles on folded fingers when fist detected
                if stable_fist:
                    for i, (tip_idx, mid_idx) in enumerate([(8,6), (12,10), (16,14), (20,18)]):
                        if lm.landmark[tip_idx].y > lm.landmark[mid_idx].y:
                            tip = lm.landmark[tip_idx]
                            cv2.circle(frame, (int(tip.x*frame.shape[1]), int(tip.y*frame.shape[0])), 8, (255, 0, 0), 2)
                
                cv2.imshow("Hand Control (press Q to hide)", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    try:
                        cv2.destroyWindow("Hand Control (press Q to hide)")
                    except Exception:
                        pass
                        
                return x
        except Exception as e:
            print(f"Error in hand tracking: {e}")
            return None
        
    def get_x_norm(self):
        """Return normalized x in [0,1] for index fingertip (8). If no hand, return last."""
        if not self.enabled:
            return None
            
        result = self._get_raw_x_norm()
        
        # Check if restart gesture was detected
        if result == "restart":
            return "restart"
        
        # Check if exit gesture was detected
        if result == "exit":
            return "exit"
            
        if result is None:
            return self.last_x_norm
            
        # Apply smoothing for more stable control
        self.smooth_x = self.smooth_x * 0.7 + result * 0.3
        self.last_x_norm = self.smooth_x
        return self.last_x_norm
        
    def release(self):
        if not self.enabled:
            return
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

# ---------- Particle system for effects ----------
class Particle:
    def __init__(self, x, y, color, speed, size, lifetime):
        self.x = x
        self.y = y
        self.color = color
        self.speed = speed
        self.size = size
        self.lifetime = lifetime
        self.age = 0
        
    def update(self):
        self.x += self.speed[0]
        self.y += self.speed[1]
        self.age += 1
        # Fade out over time
        alpha = 255 * (1 - self.age / self.lifetime)
        self.color = (*self.color[:3], int(alpha))
        
    def draw(self, surface):
        if self.age < self.lifetime:
            gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, self.color)

class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def emit(self, x, y, color, count=20, speed_range=(-3, 3), size_range=(2, 5), lifetime_range=(20, 40)):
        for _ in range(count):
            speed = (
                random.uniform(speed_range[0], speed_range[1]),
                random.uniform(speed_range[0], speed_range[1])
            )
            size = random.randint(size_range[0], size_range[1])
            lifetime = random.randint(lifetime_range[0], lifetime_range[1])
            self.particles.append(Particle(x, y, color, speed, size, lifetime))
            
    def update(self):
        for particle in self.particles[:]:
            particle.update()
            if particle.age >= particle.lifetime:
                self.particles.remove(particle)
                
    def draw(self, surface):
        for particle in self.particles:
            particle.draw(surface)

# ---------- Text animation system ----------
class TextAnimation:
    def __init__(self, text, font, color, x, y, animation_type="fade_in", duration=1.0, align="left"):
        self.text = text
        self.font = font
        self.color = color
        self.x = x
        self.y = y
        self.animation_type = animation_type
        self.duration = duration
        self.start_time = time.time()
        self.active = True
        self.align = align
        
    def update(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.active = False
        return self.active
        
    def draw(self, surface):
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration)
        
        text_surf = self.font.render(self.text, True, self.color)
        
        # Adjust position based on alignment
        if self.align == "center":
            x = self.x - text_surf.get_width() // 2
        elif self.align == "right":
            x = self.x - text_surf.get_width()
        else:  # left
            x = self.x
            
        if self.animation_type == "fade_in":
            alpha = int(255 * progress)
            text_surf.set_alpha(alpha)
            surface.blit(text_surf, (x, self.y))
            
        elif self.animation_type == "slide_in":
            offset = int(100 * (1 - progress))
            surface.blit(text_surf, (x - offset, self.y))
            
        elif self.animation_type == "pulse":
            scale = 1.0 + 0.2 * math.sin(progress * math.pi * 4)
            scaled_surf = pygame.transform.scale(text_surf, 
                                               (int(text_surf.get_width() * scale), 
                                                int(text_surf.get_height() * scale)))
            surface.blit(scaled_surf, (x - (scaled_surf.get_width() - text_surf.get_width()) // 2,
                                     self.y - (scaled_surf.get_height() - text_surf.get_height()) // 2))
            
        elif self.animation_type == "typewriter":
            visible_chars = int(len(self.text) * progress)
            visible_text = self.text[:visible_chars]
            visible_surf = self.font.render(visible_text, True, self.color)
            surface.blit(visible_surf, (x, self.y))

# ---------- game objects & logic ----------
class Car(pygame.sprite.Sprite):
    def __init__(self, img, x, y, is_player=False, car_stats=None, car_type=0):
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect(center=(x, y))
        self.is_player = is_player
        self.speed = 0
        self.max_speed = 10
        self.acceleration = 0.5
        self.deceleration = 0.3
        self.moving_left = False
        self.moving_right = False
        self.car_type = car_type
        self.engine_sound = None
        
        # Apply car stats if provided
        if car_stats and is_player:
            self.max_speed = 5 + car_stats["speed"]  # Scale to game values
            self.acceleration = 0.3 + (car_stats["handling"] / 20)
            self.deceleration = 0.2 + (car_stats["handling"] / 30)
        
    def move_to_x(self, target_x, max_step=8):
        # Smooth step toward target with acceleration/deceleration
        dx = target_x - self.rect.centerx
        
        if abs(dx) <= max_step:
            self.rect.centerx = target_x
            self.speed = 0
        else:
            # Accelerate when moving, decelerate when close to target
            if dx > 0:
                self.moving_right = True
                self.moving_left = False
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            else:
                self.moving_left = True
                self.moving_right = False
                self.speed = max(self.speed - self.acceleration, -self.max_speed)
                
            # Apply speed with easing
            ease_factor = min(1.0, abs(dx) / 100)
            actual_speed = self.speed * ease_factor
            
            self.rect.centerx += actual_speed
            
        # Keep within bounds
        self.rect.centerx = max(40, min(W - 40, self.rect.centerx))
        
    def upd_keyboard(self, keys):
        # Keyboard control with acceleration
        if keys[pygame.K_RIGHT]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
            self.moving_right = True
        elif keys[pygame.K_LEFT]:
            self.speed = max(self.speed - self.acceleration, -self.max_speed)
            self.moving_left = True
        else:
            # Decelerate when no key is pressed
            if self.speed > 0:
                self.speed = max(0, self.speed - self.deceleration)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.deceleration)
                
        self.rect.x += self.speed
        self.rect.x = max(40, min(W - 40, self.rect.x))

# Road rendering class for better visuals
class Road:
    def __init__(self, level=0):
        self.line_offset = 0
        self.line_height = 40
        self.line_width = 6
        self.road_width = W - 100
        self.road_x = 50
        self.level = level
        self.lighting = LEVEL_LIGHTING[min(level, len(LEVEL_LIGHTING)-1)]
        
    def update(self, speed):
        self.line_offset = (self.line_offset + speed) % self.line_height
        
    def draw(self, surface):
        # Get lighting colors based on level
        ambient = self.lighting["ambient"]
        
        # Draw road with perfect perspective
        road_top_width = self.road_width - 80
        road_bottom_width = self.road_width
        
        # Create road surface with gradient
        for y in range(H):
            # Calculate perspective factor
            factor = y / H
            
            # Calculate road width at this y position
            width = int(road_top_width + (road_bottom_width - road_top_width) * factor)
            
            # Calculate road position (centered)
            x = self.road_x + (road_bottom_width - width) // 2
            
            # Calculate road color with gradient
            base_darkness = 60
            darkness = int(base_darkness * (1 - factor * 0.3) * ambient)
            road_color = (darkness, darkness, darkness + int(10 * ambient))
            
            # Draw road segment
            pygame.draw.line(surface, road_color, (x, y), (x + width, y))
        
        # Draw road edges with perfect perspective
        edge_color = (
            int(255 * ambient),
            int(255 * ambient),
            int(255 * ambient)
        )
        
        # Left edge - smooth curve
        left_points = []
        right_points = []
        
        for y in range(0, H, 5):
            factor = y / H
            
            # Calculate edge positions
            left_x = self.road_x + 40 - (40 * factor)
            right_x = self.road_x + self.road_width - 40 + (40 * factor)
            
            left_points.append((left_x, y))
            right_points.append((right_x, y))
        
        # Draw smooth edge lines
        if len(left_points) > 1:
            pygame.draw.lines(surface, edge_color, False, left_points, 3)
            pygame.draw.lines(surface, edge_color, False, right_points, 3)
        
        # Draw rumble strips with perfect perspective
        rumble_color = (
            int(200 * ambient),
            int(100 * ambient),
            int(50 * ambient)
        )
        
        # Left rumble strips
        for y in range(0, H, 20):
            factor = y / H
            
            # Calculate position along the edge
            strip_width = int(15 * (1 - factor * 0.5))
            strip_x = self.road_x + 40 - (40 * factor) - strip_width
            
            # Draw rumble strip
            pygame.draw.rect(surface, rumble_color, (strip_x, y, strip_width, 10))
        
        # Right rumble strips
        for y in range(0, H, 20):
            factor = y / H
            
            # Calculate position along the edge
            strip_width = int(15 * (1 - factor * 0.5))
            strip_x = self.road_x + self.road_width - 40 + (40 * factor)
            
            # Draw rumble strip
            pygame.draw.rect(surface, rumble_color, (strip_x, y, strip_width, 10))
        
        # Draw dashed center line with perfect perspective
        line_color = (
            int(255 * ambient),
            int(255 * ambient),
            int(100 * ambient)
        )
        
        for y in range(-self.line_height + int(self.line_offset), H, self.line_height * 2):
            # Calculate perspective factor
            factor = y / H
            
            # Calculate line width based on perspective
            line_width = int(self.line_width * (0.5 + factor * 0.5))
            
            # Calculate line position
            line_x = W // 2 - line_width // 2
            
            # Draw line segment
            pygame.draw.rect(surface, line_color, 
                            (line_x, y, line_width, self.line_height))
        
        # Draw road surface texture
        for y in range(0, H, 10):
            for x in range(self.road_x, self.road_x + self.road_width, 20):
                if random.random() < 0.1:
                    # Calculate perspective factor
                    factor = y / H
                    
                    # Calculate darkness based on perspective
                    darkness = int(40 * (1 - factor * 0.3) * ambient)
                    
                    # Draw texture dot
                    pygame.draw.circle(surface, (darkness, darkness, darkness), (x, y), 1)
        
        # Draw perfect road walls/barriers
        wall_color = (
            int(150 * ambient),
            int(150 * ambient),
            int(150 * ambient)
        )
        
        # Wall cap color (top of the wall)
        wall_cap_color = (
            int(200 * ambient),
            int(200 * ambient),
            int(200 * ambient)
        )
        
        # Left wall with perfect perspective
        for y in range(0, H, 5):
            factor = y / H
            
            # Calculate wall dimensions
            wall_height = int(25 * (1 - factor * 0.6))
            wall_x = self.road_x + 40 - (40 * factor) - wall_height
            
            # Draw wall segment
            pygame.draw.rect(surface, wall_color, (wall_x, y, wall_height, 5))
            
            # Draw wall cap
            if y % 10 < 5:
                pygame.draw.rect(surface, wall_cap_color, (wall_x, y, wall_height, 2))
        
        # Right wall with perfect perspective
        for y in range(0, H, 5):
            factor = y / H
            
            # Calculate wall dimensions
            wall_height = int(25 * (1 - factor * 0.6))
            wall_x = self.road_x + self.road_width - 40 + (40 * factor)
            
            # Draw wall segment
            pygame.draw.rect(surface, wall_color, (wall_x, y, wall_height, 5))
            
            # Draw wall cap
            if y % 10 < 5:
                pygame.draw.rect(surface, wall_cap_color, (wall_x, y, wall_height, 2))
        
        # Draw reflective strips on walls
        strip_color = (
            int(255 * ambient),
            int(255 * ambient),
            int(255 * ambient)
        )
        
        # Left wall reflective strips
        for y in range(0, H, 20):
            factor = y / H
            
            # Calculate strip position
            strip_height = int(25 * (1 - factor * 0.6))
            strip_x = self.road_x + 40 - (40 * factor) - strip_height
            
            # Draw reflective strip
            if y % 40 < 20:
                pygame.draw.rect(surface, strip_color, (strip_x + strip_height - 3, y, 3, 10))
        
        # Right wall reflective strips
        for y in range(0, H, 20):
            factor = y / H
            
            # Calculate strip position
            strip_height = int(25 * (1 - factor * 0.6))
            strip_x = self.road_x + self.road_width - 40 + (40 * factor)
            
            # Draw reflective strip
            if y % 40 < 20:
                pygame.draw.rect(surface, strip_color, (strip_x, y, 3, 10))

# Background scenery
class Background:
    def __init__(self, level=0):
        self.trees = []
        self.buildings = []
        self.clouds = []
        self.level = level
        self.lighting = LEVEL_LIGHTING[min(level, len(LEVEL_LIGHTING)-1)]
        self.generate_scenery()
        
    def generate_scenery(self):
        # Generate trees
        for _ in range(20):
            x = random.randint(0, W)
            y = random.randint(0, H)
            size = random.randint(20, 40)
            self.trees.append((x, y, size))
            
        # Generate buildings
        for _ in range(10):
            x = random.randint(0, W)
            width = random.randint(40, 80)
            height = random.randint(100, 200)
            self.buildings.append((x, H, width, height))
            
        # Generate clouds
        for _ in range(5):
            x = random.randint(0, W)
            y = random.randint(0, H // 3)
            size = random.randint(30, 60)
            self.clouds.append((x, y, size))
            
    def update(self, speed):
        # Move scenery based on speed
        move_speed = speed * 0.2
        
        # Update trees
        new_trees = []
        for x, y, size in self.trees:
            new_y = (y + move_speed) % H
            new_trees.append((x, new_y, size))
        self.trees = new_trees
        
        # Update buildings
        new_buildings = []
        for x, y, width, height in self.buildings:
            new_y = (y + move_speed) % (H + height)
            new_buildings.append((x, new_y, width, height))
        self.buildings = new_buildings
        
        # Update clouds (slower movement)
        new_clouds = []
        for x, y, size in self.clouds:
            new_y = (y + move_speed * 0.5) % (H // 2)
            new_x = (x + move_speed * 0.1) % W
            new_clouds.append((new_x, new_y, size))
        self.clouds = new_clouds
        
    def draw(self, surface):
        # Draw sky gradient based on lighting
        sky_color = self.lighting["sky"]
        horizon_color = self.lighting["horizon"]
        
        for y in range(H // 2):
            # Interpolate between sky and horizon colors
            factor = y / (H // 2)
            r = int(sky_color[0] * (1 - factor) + horizon_color[0] * factor)
            g = int(sky_color[1] * (1 - factor) + horizon_color[1] * factor)
            b = int(sky_color[2] * (1 - factor) + horizon_color[2] * factor)
            pygame.draw.line(surface, (r, g, b), (0, y), (W, y))
            
        # Draw stars only in dark lighting
        if self.lighting["name"] == "dark":
            random.seed(42)  # Fixed seed for consistent stars
            for _ in range(100):
                x = random.randint(0, W)
                y = random.randint(0, H // 2)
                size = random.randint(1, 2)
                brightness = int(255 * self.lighting["ambient"])
                pygame.draw.circle(surface, (brightness, brightness, brightness), (x, y), size)
            random.seed()  # Reset seed
        
        # Draw moon/sun based on lighting
        if self.lighting["name"] == "dark":
            # Draw moon
            pygame.draw.circle(surface, (240, 240, 255), (W - 100, 80), 30)
            pygame.draw.circle(surface, (200, 200, 220), (W - 90, 70), 25)
        elif self.lighting["name"] == "light":
            # Draw sun
            pygame.draw.circle(surface, (255, 255, 200), (W - 100, 80), 40)
            # Sun rays
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x1 = W - 100 + 40 * math.cos(rad)
                y1 = 80 + 40 * math.sin(rad)
                x2 = W - 100 + 60 * math.cos(rad)
                y2 = 80 + 60 * math.sin(rad)
                pygame.draw.line(surface, (255, 255, 200), (x1, y1), (x2, y2), 3)
        else:
            # Draw setting sun
            pygame.draw.circle(surface, (255, 180, 100), (W - 100, 120), 35)
            
        # Draw clouds
        for x, y, size in self.clouds:
            cloud_color = (
                int(200 * self.lighting["ambient"]),
                int(200 * self.lighting["ambient"]),
                int(220 * self.lighting["ambient"])
            )
            pygame.draw.ellipse(surface, cloud_color, (x, y, size * 2, size))
            
        # Draw buildings
        for x, y, width, height in self.buildings:
            if y - height < H:  # Only draw if visible
                # Building body with gradient
                for i in range(height):
                    factor = 1 - (i / height) * 0.3
                    color = tuple(int(c * factor * self.lighting["ambient"]) for c in (40, 40, 60))
                    pygame.draw.line(surface, color, (x, y - i), (x + width, y - i))
                    
                # Windows
                for wx in range(x + 5, x + width - 5, 10):
                    for wy in range(int(y - height + 10), int(y - 10), 15):
                        if random.random() > 0.3:  # Randomly lit windows
                            window_color = (
                                int(255 * self.lighting["ambient"]),
                                int(255 * self.lighting["ambient"]),
                                int(200 * self.lighting["ambient"])
                            )
                            pygame.draw.rect(surface, window_color, (wx, wy, 6, 8))
                            
        # Draw trees
        for x, y, size in self.trees:
            if y < H:  # Only draw if visible
                # Tree trunk
                trunk_color = (
                    int(30 * self.lighting["ambient"]),
                    int(20 * self.lighting["ambient"]),
                    int(10 * self.lighting["ambient"])
                )
                pygame.draw.rect(surface, trunk_color, (x - size // 10, y - size // 2, size // 5, size // 2))
                # Tree leaves
                leaf_color = (
                    int(10 * self.lighting["ambient"]),
                    int(30 * self.lighting["ambient"]),
                    int(10 * self.lighting["ambient"])
                )
                pygame.draw.circle(surface, leaf_color, (x, y - size // 2), size // 2)

LANES = {"L": W // 4, "C": W // 2, "R": 3 * W // 4}
LANE_GRAPH = {
    "L": {"C": 1},
    "C": {"L": 1, "R": 1},
    "R": {"C": 1}
}

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}
    unvisited = set(graph.keys())
    while unvisited:
        current = min(unvisited, key=lambda node: dist[node])
        unvisited.remove(current)
        for neighbor, cost in graph[current].items():
            alt = dist[current] + cost
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = current
    return dist, prev

def get_closest_lane(x):
    return min(LANES, key=lambda l: abs(LANES[l] - x))

def choose_enemy_lane(player_x, last_enemy_lane=None):
    player_lane = get_closest_lane(player_x)
    lanes = list(LANES.keys())
    if last_enemy_lane not in lanes:
        return random.choice(lanes)
    dist, prev = dijkstra(LANE_GRAPH, last_enemy_lane)
    path, cur = [], player_lane
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    next_lane = path[1] if len(path) > 1 else player_lane
    if random.random() < 0.1:
        choices = [next_lane] + list(LANE_GRAPH[next_lane].keys())
        next_lane = random.choice(choices)
    return next_lane

def draw_finish_line():
    line_height, line_width, y = 10, W - 100, 100
    # Draw checkered pattern with glow effect
    square_size = 20
    for x in range(50, 50 + line_width, square_size):
        for y_pos in range(y, y + line_height, square_size):
            if (x // square_size + y_pos // square_size) % 2 == 0:
                pygame.draw.rect(screen, (255, 255, 255), (x, y_pos, square_size, square_size))
            else:
                pygame.draw.rect(screen, (0, 0, 0), (x, y_pos, square_size, square_size))
    
    # Add glow effect
    glow_surf = pygame.Surface((line_width, line_height + 20), pygame.SRCALPHA)
    for i in range(10):
        alpha = 100 - i * 10
        pygame.draw.rect(glow_surf, (255, 215, 0, alpha), (0, i, line_width, line_height + 20 - 2*i), border_radius=5)
    screen.blit(glow_surf, (50, y - 10))
    
    # Draw finish text with animation
    font = deadcrt_font
    text = font.render("FINISH", True, (255, 215, 0))
    text_rect = text.get_rect(center=(W//2, y - 60))
    
    # Pulsing effect
    scale = 1.0 + 0.1 * math.sin(time.time() * 5)
    scaled_text = pygame.transform.scale(text, (int(text.get_width() * scale), int(text.get_height() * scale)))
    screen.blit(scaled_text, (text_rect.centerx - scaled_text.get_width() // 2, 
                             text_rect.centery - scaled_text.get_height() // 2))

def draw_hud(level, score, high_score, player_name, show_restart_hint=True):
    # Draw HUD background panel
    panel = pygame.Surface((200, 160), pygame.SRCALPHA)
    panel.fill((30, 30, 45, 200))
    screen.blit(panel, (10, 10))
    
    # Draw level indicator
    level_text = small_sharpshooter.render(f"LEVEL: {level}", True, HIGHLIGHT_COLOR)
    screen.blit(level_text, (20, 20))
    
    # Draw score
    score_text = small_dynatecha.render(f"SCORE: {score}", True, TEXT_COLOR)
    screen.blit(score_text, (20, 50))
    
    # Draw high score
    high_score_text = small_dynatecha.render(f"HIGH: {high_score}", True, HIGHLIGHT_COLOR)
    screen.blit(high_score_text, (20, 80))
    
    # Draw player name
    name_text = small_future.render(f"{player_name}", True, ACCENT_COLOR)
    screen.blit(name_text, (20, 110))
    
    # Draw control indicator
    ctrl = "HAND" if USE_HANDS else "KEYBOARD"
    ctrl_text = small_gomarice.render(f"CTRL: {ctrl}", True, TEXT_COLOR)
    screen.blit(ctrl_text, (W - ctrl_text.get_width() - 10, 10))
    
    # Draw restart gesture indicator
    if USE_HANDS and show_restart_hint:
        gesture_text = small_dynatecha.render("PINCH: Restart", True, WARNING_COLOR)
        screen.blit(gesture_text, (W - gesture_text.get_width() - 10, 40))
        
        exit_text = small_dynatecha.render("FIST: Exit", True, WARNING_COLOR)
        screen.blit(exit_text, (W - exit_text.get_width() - 10, 70))
    
    # Draw progress bar for level completion
    progress = min(1.0, score / (400 * level))
    bar_width = 200
    bar_height = 10
    bar_x = W - bar_width - 10
    bar_y = 100
    
    # Background
    pygame.draw.rect(screen, (50, 50, 60), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
    # Progress with gradient
    for i in range(int(bar_width * progress)):
        color_factor = i / (bar_width * progress)
        color = (
            int(255 * color_factor),
            int(255 * (1 - color_factor)),
            0
        )
        pygame.draw.line(screen, color, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height))
    # Border
    pygame.draw.rect(screen, (100, 100, 120), (bar_x, bar_y, bar_width, bar_height), 1, border_radius=5)

def run(level, hands: HandIndexTracker, high_score, player_name, selected_car_index):
    speed = 4 + 2 * level
    player = Car(PLAYER_CARS[selected_car_index], W // 2, H - 80, is_player=True, car_stats=car_stats[selected_car_index], car_type=selected_car_index)
    enemies = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group(player)
    score, spawn_timer = 0, 0
    last_enemy_lane = None
    
    # Create game objects with level-specific lighting
    road = Road(level-1)  # level-1 because level starts at 1 but list index starts at 0
    background = Background(level-1)
    particles = ParticleSystem()
    text_animations = []
    
    # Load and play engine sound for selected car
    engine_sound = ENGINE_SOUNDS[selected_car_index]
    if engine_sound:
        engine_channel = pygame.mixer.Channel(1)
        engine_channel.play(engine_sound, loops=-1)
        engine_channel.set_volume(0.3)
    
    # Add level start animation
    text_animations.append(TextAnimation(
        f"LEVEL {level}", 
        deadcrt_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        H//2 - 50, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Calibrate hand tracking if enabled
    if hands and hands.enabled and not hands.calibrated:
        text_animations.append(TextAnimation(
            "CALIBRATING HAND CONTROL...", 
            small_future, 
            WARNING_COLOR, 
            W//2, 
            H//2, 
            "typewriter", 
            3.0,
            "center"
        ))
        hands.calibrate()
    
    # Game loop
    running = True
    paused = False
    game_over = False
    level_complete = False
    restart_feedback_time = 0
    
    while running and not game_over and not level_complete:
        # Handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return False to exit the game
                return "exit", score
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    paused = not paused
                    if paused:
                        text_animations.append(TextAnimation(
                            "PAUSED", 
                            deadcrt_font, 
                            WARNING_COLOR, 
                            W//2, 
                            H//2, 
                            "fade_in", 
                            0.5,
                            "center"
                        ))
                elif e.key == pygame.K_ESCAPE:
                    return "exit", score
                elif e.key == pygame.K_r:  # Keyboard restart
                    if engine_sound:
                        engine_channel.stop()
                    return "restart", score  # Return with restart flag
                    
        if paused:
            # Draw pause screen
            pause_surf = pygame.Surface((W, H), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 150))
            screen.blit(pause_surf, (0, 0))
            
            continue_text = small_dynatecha.render("Press P to continue", True, TEXT_COLOR)
            continue_rect = continue_text.get_rect(center=(W//2, H//2 + 50))
            screen.blit(continue_text, continue_rect)
            pygame.display.flip()
            clock.tick(30)
            continue
            
        # ---- control: hand first, keyboard fallback ----
        target_x = None
        if hands and hands.enabled:
            hand_result = hands.get_x_norm()  # Can be x position, "restart", or "exit"
            
            if hand_result == "exit":  # Exit gesture detected
                # Show exit feedback
                text_animations.append(TextAnimation(
                    "EXIT GESTURE DETECTED!", 
                    deadcrt_font, 
                    WARNING_COLOR, 
                    W//2, 
                    H//2, 
                    "pulse", 
                    1.0,
                    "center"
                ))
                if snd_select:
                    snd_select.play()
                if engine_sound:
                    engine_channel.stop()
                return "exit", score  # Return with exit flag
            elif hand_result == "restart":  # Restart gesture detected
                # Show restart feedback
                restart_feedback_time = time.time()
                text_animations.append(TextAnimation(
                    "RESTART GESTURE DETECTED!", 
                    deadcrt_font, 
                    WARNING_COLOR, 
                    W//2, 
                    H//2, 
                    "pulse", 
                    1.0,
                    "center"
                ))
                if snd_select:
                    snd_select.play()
            elif hand_result is not None:
                margin = 40
                target_x = int(margin + hand_result * (W - 2*margin))
                
        keys = pygame.key.get_pressed()
        if target_x is not None:
            player.move_to_x(target_x, max_step=10)
        else:
            player.upd_keyboard(keys)
            
        # Update engine sound based on player speed
        if engine_sound and engine_channel:
            # Adjust pitch based on speed
            speed_factor = min(1.0, abs(player.speed) / player.max_speed)
            engine_channel.set_volume(0.2 + 0.3 * speed_factor)
            
        # Update game objects
        road.update(speed)
        background.update(speed)
        particles.update()
        
        # Update text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
        
        # Check if restart was requested and enough time has passed for feedback
        if restart_feedback_time > 0 and time.time() - restart_feedback_time > 0.5:
            if engine_sound:
                engine_channel.stop()
            return "restart", score  # Return with restart flag
        
        # Spawn enemies
        spawn_timer += 1
        spawn_rate = max(20, 50 - level * 5)  # Increase spawn rate with level
        
        if spawn_timer > spawn_rate:
            spawn_timer = 0
            lane_key = choose_enemy_lane(player.rect.centerx, last_enemy_lane)
            last_enemy_lane = lane_key
            lane_x = LANES[lane_key]
            enemy_img = random.choice(ENEMY_IMGS)
            enemy = Car(enemy_img, lane_x, -60)
            enemies.add(enemy)
            all_sprites.add(enemy)
            
        # Update enemies
        for enemy in list(enemies):
            enemy.rect.y += speed
            
            # Add slight horizontal movement to enemies
            if random.random() < 0.01:
                lane_key = random.choice(list(LANES.keys()))
                target_x = LANES[lane_key]
                enemy.move_to_x(target_x, max_step=2)
                
            if enemy.rect.top > H:
                enemies.remove(enemy)
                all_sprites.remove(enemy)
                
        # Check collisions - crash sound plays and restarts game
        if pygame.sprite.spritecollide(player, enemies, False):
            snd_crash.play()
            
            # Create explosion particles
            particles.emit(player.rect.centerx, player.rect.centery, (255, 100, 0), 30, 
                          speed_range=(-5, 5), size_range=(3, 8), lifetime_range=(20, 40))
            
            # Add game over text animation
            text_animations.append(TextAnimation(
                "CRASH!", 
                deadcrt_font, 
                WARNING_COLOR, 
                W//2, 
                H//2 - 50, 
                "pulse", 
                1.0,
                "center"
            ))
            
            game_over = True
            
        score += 1
        
        # Check level completion
        if score > 400 * level:
            snd_beep.play()
            
            # Add level complete animation
            text_animations.append(TextAnimation(
                "LEVEL COMPLETE!", 
                future_font, 
                HIGHLIGHT_COLOR, 
                W//2, 
                H//2 - 50, 
                "slide_in", 
                1.5,
                "center"
            ))
            
            level_complete = True
            
        # Draw everything
        # Background
        background.draw(screen)
        
        # Road
        road.draw(screen)
        
        # Sprites
        all_sprites.draw(screen)
        
        # Particles
        particles.draw(screen)
        
        # HUD
        draw_hud(level, score, high_score, player_name, show_restart_hint=(restart_feedback_time == 0))
        
        # Show finish line when close to completing level
        if score > 350 * level:
            draw_finish_line()
            
        # Draw text animations
        for anim in text_animations:
            anim.draw(screen)
            
        pygame.display.flip()
        clock.tick(60)
        
    # Stop engine sound when level ends
    if engine_sound and engine_channel:
        engine_channel.stop()
        
    if level_complete:
        return "completed", score
    else:
        return "restart", score

def game_over_screen(score, high_score, hands, player_name):
    # Update high score if needed
    if score > high_score:
        high_score = score
        
    text_animations = []
    
    # Add game over animation
    text_animations.append(TextAnimation(
        "GAME OVER", 
        deadcrt_font, 
        WARNING_COLOR, 
        W//2, 
        H//2 - 100, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add player name animation
    text_animations.append(TextAnimation(
        f"PLAYER: {player_name}", 
        small_future, 
        ACCENT_COLOR, 
        W//2, 
        H//2 - 50, 
        "slide_in", 
        1.0,
        "center"
    ))
    
    # Add score animation
    text_animations.append(TextAnimation(
        f"SCORE: {score}", 
        small_sharpshooter, 
        TEXT_COLOR, 
        W//2, 
        H//2, 
        "fade_in", 
        1.0,
        "center"
    ))
    
    # Add high score animation
    if score >= high_score:
        text_animations.append(TextAnimation(
            "NEW HIGH SCORE!", 
            small_gomarice, 
            HIGHLIGHT_COLOR, 
            W//2, 
            H//2 + 50, 
            "typewriter", 
            1.5,
            "center"
        ))
    else:
        text_animations.append(TextAnimation(
            f"HIGH: {high_score}", 
            small_dynatecha, 
            HIGHLIGHT_COLOR, 
            W//2, 
            H//2 + 50, 
            "fade_in", 
            1.0,
            "center"
        ))
    
    # Add restart instruction
    text_animations.append(TextAnimation(
        "Press R to Restart or ESC to Quit", 
        small_dynatecha, 
        TEXT_COLOR, 
        W//2, 
        H//2 + 100, 
        "typewriter", 
        2.0,
        "center"
    ))
    
    if USE_HANDS:
        text_animations.append(TextAnimation(
            "Or use PINCH to restart or FIST to exit", 
            small_dynatecha, 
            WARNING_COLOR, 
            W//2, 
            H//2 + 130, 
            "typewriter", 
            2.0,
            "center"
        ))
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return False to exit the game
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    return True  # Return restart flag
                elif e.key == pygame.K_ESCAPE:
                    return False  # Return no restart
                    
        # Check for hand gesture restart
        if hands and hands.enabled:
            hand_result = hands.get_x_norm()
            if hand_result == "restart":
                return True  # Return restart flag
            elif hand_result == "exit":
                return False  # Return no restart
                    
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        # Update and draw text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
            else:
                anim.draw(screen)
                
        pygame.display.flip()
        clock.tick(60)

def victory_screen(total_score, high_score, hands, player_name):
    # Update high score if needed
    if total_score > high_score:
        high_score = total_score
        
    text_animations = []
    
    # Add victory animation
    text_animations.append(TextAnimation(
        "VICTORY!", 
        deadcrt_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        H//2 - 100, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add player name animation
    text_animations.append(TextAnimation(
        f"PLAYER: {player_name}", 
        small_future, 
        ACCENT_COLOR, 
        W//2, 
        H//2 - 50, 
        "slide_in", 
        1.0,
        "center"
    ))
    
    # Add score animation
    text_animations.append(TextAnimation(
        f"FINAL SCORE: {total_score}", 
        small_sharpshooter, 
        TEXT_COLOR, 
        W//2, 
        H//2, 
        "fade_in", 
        1.0,
        "center"
    ))
    
    # Add high score animation
    if total_score >= high_score:
        text_animations.append(TextAnimation(
            "NEW HIGH SCORE!", 
            small_gomarice, 
            HIGHLIGHT_COLOR, 
            W//2, 
            H//2 + 50, 
            "typewriter", 
            1.5,
            "center"
        ))
    else:
        text_animations.append(TextAnimation(
            f"HIGH: {high_score}", 
            small_dynatecha, 
            HIGHLIGHT_COLOR, 
            W//2, 
            H//2 + 50, 
            "fade_in", 
            1.0,
            "center"
        ))
    
    # Add continue instruction
    text_animations.append(TextAnimation(
        "Press C to continue or ESC to Quit", 
        small_dynatecha, 
        TEXT_COLOR, 
        W//2, 
        H//2 + 100, 
        "typewriter", 
        2.0,
        "center"
    ))
    
    # Create celebration particles
    particles = ParticleSystem()
    for _ in range(5):
        x = random.randint(50, W - 50)
        y = random.randint(50, H // 2)
        color = random.choice([(255, 215, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)])
        particles.emit(x, y, color, 50, speed_range=(-3, 3), size_range=(2, 6), lifetime_range=(30, 60))
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return False to exit the game
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_c:
                    return True  # Return continue flag
                elif e.key == pygame.K_ESCAPE:
                    return False  # Return no continue
                    
        # Check for hand gesture exit
        if hands and hands.enabled and hands.get_x_norm() == "exit":
            return False  # Return no continue
                    
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        # Update and draw particles
        particles.update()
        particles.draw(screen)
        
        # Update and draw text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
            else:
                anim.draw(screen)
                
        pygame.display.flip()
        clock.tick(60)

def end_story_screen(hands):
    # Release the camera if it exists - CAMERA DISABLED
    if hands and hands.enabled:
        hands.release()
    
    # Play background music if available
    try:
        pygame.mixer.music.load("sounds/music.wav")
        pygame.mixer.music.play(-1)  # Loop indefinitely
        pygame.mixer.music.set_volume(0.5)
    except:
        pass
    
    text_animations = []
    
    # Add title animation
    text_animations.append(TextAnimation(
        "THE JOURNEY", 
        deadcrt_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        50, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add story text animations with typewriter effect
    story_lines = [
        "The road was long and challenging,",
        "filled with obstacles at every turn.",
        "Yet through skill and determination,",
        "you emerged victorious against all odds.",
        "",
        "The background music, 'music.wav',",
        "was a constant companion throughout your journey.",
        "Its rhythm guided your movements,",
        "its melody fueled your spirit,",
        "and its harmony became the soundtrack",
        "to your incredible victory.",
        "",
        "Story by: Abhishek Gupta",
        "",
        "Thank you for playing!"
    ]
    
    y_pos = 120
    for i, line in enumerate(story_lines):
        if line:  # Skip empty lines
            text_animations.append(TextAnimation(
                line,
                small_dynatecha,
                STORY_TEXT_COLOR,
                W//2,  # Center position
                y_pos,
                "typewriter",
                1.0 + i * 0.15,  # Faster animations to fit in 10 seconds
                "center"  # Center alignment
            ))
        y_pos += 25
    
    # Create ambient particles
    particles = ParticleSystem()
    
    # Track when to show the end message
    show_end_message = False
    end_message_start_time = 0
    story_complete_time = 10  # Time in seconds when story is considered complete (changed to 10)
    
    start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return False to exit the game
                pygame.mixer.music.stop()
                return False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop()
                    return False  # Exit game
                # Allow any key to skip the story
                if not show_end_message:
                    show_end_message = True
                    end_message_start_time = current_time
                    
        # Check if it's time to show the end message
        if elapsed >= story_complete_time and not show_end_message:
            show_end_message = True
            end_message_start_time = current_time
        
        # Occasionally emit particles
        if random.random() < 0.05 and not show_end_message:
            x = random.randint(50, W - 50)
            y = random.randint(50, H // 2)
            color = random.choice([(255, 215, 0), (100, 100, 255), (255, 100, 255)])
            particles.emit(x, y, color, 10, speed_range=(-1, 1), size_range=(2, 4), lifetime_range=(20, 40))
        
        # Draw story background
        screen.fill(STORY_BACKGROUND)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        if not show_end_message:
            # Update and draw particles
            particles.update()
            particles.draw(screen)
            
            # Update and draw text animations
            for anim in text_animations[:]:
                if not anim.update():
                    text_animations.remove(anim)
                else:
                    anim.draw(screen)
                    
            # Add skip instruction
            skip_text = small_dynatecha.render("Press any key to skip", True, ACCENT_COLOR)
            skip_rect = skip_text.get_rect(center=(W//2, H - 30))
            screen.blit(skip_text, skip_rect)
        else:
            # Show the end message
            message = "Game Ended! \n Next Level Coming Soon!"
            lines = message.split('\n')
            y = H // 2 - 40
            for line in lines:
                text = future_font.render(line, True, WARNING_COLOR)
                text_rect = text.get_rect(center=(W//2, y))
                screen.blit(text, text_rect)
                y += 50
            
            # Check if it's time to restart automatically (after 3 seconds instead of 5)
            if current_time - end_message_start_time > 3:  # Show message for 3 seconds
                pygame.mixer.music.stop()
                return True  # Auto restart
                
        pygame.display.flip()
        clock.tick(60)

def hand_recognition_screen():
    # Reinitialize hand tracking - CAMERA ENABLED
    hands = HandIndexTracker() if USE_HANDS else None
    
    if hands and hands.enabled:
        hands.calibrate()
    
    text_animations = []
    
    # Add title animation
    text_animations.append(TextAnimation(
        "HAND RECOGNITION", 
        deadcrt_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        50, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add instructions animation
    text_animations.append(TextAnimation(
        "Pinch gesture: Restart the game", 
        small_dynatecha, 
        TEXT_COLOR, 
        W//2, 
        H//2 - 50, 
        "typewriter", 
        2.0,
        "center"
    ))
    
    # Add exit instruction
    text_animations.append(TextAnimation(
        "Closed fist gesture: Exit the game", 
        small_dynatecha, 
        WARNING_COLOR, 
        W//2, 
        H//2, 
        "typewriter", 
        2.0,
        "center"
    ))
    
    # Add waiting message
    text_animations.append(TextAnimation(
        "Waiting for gesture...", 
        small_future, 
        ACCENT_COLOR, 
        W//2, 
        H//2 + 50, 
        "fade_in", 
        1.0,
        "center"
    ))
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                if hands:
                    hands.release()
                return False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if hands:
                        hands.release()
                    return False
                elif e.key == pygame.K_r:
                    if hands:
                        hands.release()
                    return True
                    
        # Check for hand gesture restart
        if hands and hands.enabled:
            hand_result = hands.get_x_norm()
            if hand_result == "restart":
                hands.release()
                return True  # Return restart flag
            elif hand_result == "exit":
                hands.release()
                return False  # Return exit flag
                    
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        # Update and draw text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
            else:
                anim.draw(screen)
                
        pygame.display.flip()
        clock.tick(60)

def car_selection_screen():
    selected_car = 0
    car_rects = []
    
    # Create car selection grid (5x2)
    car_width = 120
    car_height = 100
    spacing = 20
    start_x = (W - (5 * car_width + 4 * spacing)) // 2
    start_y = 150
    
    for i in range(10):
        row = i // 5
        col = i % 5
        x = start_x + col * (car_width + spacing)
        y = start_y + row * (car_height + spacing)
        car_rects.append(pygame.Rect(x, y, car_width, car_height))
    
    text_animations = []
    
    # Add title animation
    text_animations.append(TextAnimation(
        "SELECT YOUR CAR", 
        gomarice_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        50, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add instructions animation
    text_animations.append(TextAnimation(
        "Use arrow keys or mouse to select, press ENTER to confirm", 
        small_dynatecha, 
        TEXT_COLOR, 
        W//2, 
        100, 
        "typewriter", 
        2.0,
        "center"
    ))
    
    # Create a channel for previewing car sounds
    preview_channel = pygame.mixer.Channel(2)
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return -1 to exit the game
                return -1
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_LEFT:
                    selected_car = (selected_car - 1) % 10
                    snd_select.play()
                    # Preview engine sound
                    if ENGINE_SOUNDS[selected_car]:
                        preview_channel.stop()
                        preview_channel.play(ENGINE_SOUNDS[selected_car])
                elif e.key == pygame.K_RIGHT:
                    selected_car = (selected_car + 1) % 10
                    snd_select.play()
                    # Preview engine sound
                    if ENGINE_SOUNDS[selected_car]:
                        preview_channel.stop()
                        preview_channel.play(ENGINE_SOUNDS[selected_car])
                elif e.key == pygame.K_UP:
                    selected_car = (selected_car - 5) % 10
                    snd_select.play()
                    # Preview engine sound
                    if ENGINE_SOUNDS[selected_car]:
                        preview_channel.stop()
                        preview_channel.play(ENGINE_SOUNDS[selected_car])
                elif e.key == pygame.K_DOWN:
                    selected_car = (selected_car + 5) % 10
                    snd_select.play()
                    # Preview engine sound
                    if ENGINE_SOUNDS[selected_car]:
                        preview_channel.stop()
                        preview_channel.play(ENGINE_SOUNDS[selected_car])
                elif e.key == pygame.K_RETURN:
                    preview_channel.stop()
                    return selected_car
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, rect in enumerate(car_rects):
                    if rect.collidepoint(mouse_pos):
                        selected_car = i
                        snd_select.play()
                        # Preview engine sound
                        if ENGINE_SOUNDS[selected_car]:
                            preview_channel.stop()
                            preview_channel.play(ENGINE_SOUNDS[selected_car])
                        return selected_car
        
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        # Draw cars and info
        for i, rect in enumerate(car_rects):
            # Draw car background
            if i == selected_car:
                # Glow effect for selected car
                glow_surf = pygame.Surface((car_width + 20, car_height + 20), pygame.SRCALPHA)
                for j in range(10):
                    alpha = 100 - j * 10
                    pygame.draw.rect(glow_surf, (100, 100, 255, alpha), 
                                    (j, j, car_width + 20 - 2*j, car_height + 20 - 2*j), border_radius=10)
                screen.blit(glow_surf, (rect.x - 10, rect.y - 10))
                
                pygame.draw.rect(screen, PANEL_COLOR, rect.inflate(10, 10), border_radius=10)
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect.inflate(10, 10), 3, border_radius=10)
            else:
                pygame.draw.rect(screen, PANEL_COLOR, rect, border_radius=5)
            
            # Draw car
            car_img = pygame.transform.scale(PLAYER_CARS[i], (car_width - 20, car_height - 20))
            screen.blit(car_img, (rect.x + 10, rect.y + 10))
            
            # Draw car name with proper size to avoid overlapping
            name_text = small_dynatecha.render(car_names[i], True, TEXT_COLOR)
            name_rect = name_text.get_rect(center=(rect.centerx, rect.bottom + 15))
            screen.blit(name_text, name_rect)
        
        # Draw selected car info panel
        info_rect = pygame.Rect(W // 2 - 200, 400, 400, 150)
        pygame.draw.rect(screen, PANEL_COLOR, info_rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT_COLOR, info_rect, 2, border_radius=10)
        
        # Car name with proper sizing
        name_text = future_font.render(car_names[selected_car], True, HIGHLIGHT_COLOR)
        name_rect = name_text.get_rect(center=(info_rect.centerx, info_rect.y + 20))
        screen.blit(name_text, name_rect)
        
        # Car stats with proper spacing
        stats = car_stats[selected_car]
        speed_text = small_dynatecha.render(f"Speed: {'★' * stats['speed']}", True, TEXT_COLOR)
        handling_text = small_dynatecha.render(f"Handling: {'★' * stats['handling']}", True, TEXT_COLOR)
        desc_text = small_dynatecha.render(stats["description"], True, TEXT_COLOR)
        
        screen.blit(speed_text, (info_rect.x + 20, info_rect.y + 60))
        screen.blit(handling_text, (info_rect.x + 20, info_rect.y + 90))
        desc_rect = desc_text.get_rect(center=(info_rect.centerx, info_rect.y + 120))
        screen.blit(desc_text, desc_rect)
        
        # Update and draw text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
            else:
                anim.draw(screen)
                
        pygame.display.flip()
        clock.tick(60)

def name_input_screen(selected_car_index):
    player_name = ""
    active = True
    
    text_animations = []
    
    # Add title animation
    text_animations.append(TextAnimation(
        "ENTER YOUR NAME", 
        gomarice_font, 
        HIGHLIGHT_COLOR, 
        W//2, 
        50, 
        "pulse", 
        2.0,
        "center"
    ))
    
    # Add instructions animation
    text_animations.append(TextAnimation(
        "Press ENTER to continue", 
        small_dynatecha, 
        TEXT_COLOR, 
        W//2, 
        410, 
        "typewriter", 
        1.5,
        "center"
    ))
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                # Don't release hands here, just return None to exit the game
                return None
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    if player_name:  # Only accept non-empty names
                        return player_name
                elif e.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]
                else:
                    # Only allow letters, numbers, and spaces
                    if e.unicode.isalnum() or e.unicode == ' ':
                        if len(player_name) < 15:  # Limit name length
                            player_name += e.unicode
        
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw grid pattern for visual interest
        for x in range(0, W, 40):
            pygame.draw.line(screen, (25, 25, 35), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(screen, (25, 25, 35), (0, y), (W, y))
        
        # Draw selected car
        car_img = pygame.transform.scale(PLAYER_CARS[selected_car_index], (150, 120))
        car_rect = car_img.get_rect(center=(W//2, 240))
        screen.blit(car_img, car_rect)
        
        # Draw car name
        car_name_text = future_font.render(car_names[selected_car_index], True, HIGHLIGHT_COLOR)
        car_name_rect = car_name_text.get_rect(center=(W//2, 320))
        screen.blit(car_name_text, car_name_rect)
        
        # Draw input box
        input_box = pygame.Rect(W // 2 - 150, 350, 300, 40)
        pygame.draw.rect(screen, PANEL_COLOR, input_box, border_radius=5)
        pygame.draw.rect(screen, ACCENT_COLOR if active else (70, 70, 80), input_box, 2, border_radius=5)
        
        # Draw player name
        name_surface = small_dynatecha.render(player_name, True, TEXT_COLOR)
        name_rect = name_surface.get_rect(center=(input_box.centerx, input_box.centery))
        screen.blit(name_surface, name_rect)
        
        # Draw cursor
        if active and time.time() % 1 < 0.5:
            cursor_x = input_box.centerx + name_surface.get_width() // 2 + 5
            pygame.draw.line(screen, HIGHLIGHT_COLOR, (cursor_x, input_box.y + 10), 
                            (cursor_x, input_box.y + 30), 2)
        
        # Update and draw text animations
        for anim in text_animations[:]:
            if not anim.update():
                text_animations.remove(anim)
            else:
                anim.draw(screen)
                
        pygame.display.flip()
        clock.tick(60)

def main():
    global USE_HANDS
    
    # Check for OpenCV and MediaPipe availability
    if not USE_HANDS:
        print("Hand tracking disabled: OpenCV/MediaPipe not available")
    else:
        print("Checking camera availability...")
        if not check_camera_available():
            print("No camera detected or camera not accessible. Disabling hand tracking.")
            USE_HANDS = False
        else:
            print("Camera detected. Hand tracking enabled.")
    
    # Load high score
    high_score = 0
    try:
        with open("highscore.txt", "r") as f:
            high_score = int(f.read())
    except:
        pass
        
    # Play background music if available
    if snd_background:
        try:
            pygame.mixer.music.load("sounds/background.wav")
            pygame.mixer.music.play(-1)  # Loop indefinitely
            pygame.mixer.music.set_volume(0.3)
        except:
            pass
    
    # Game loop with restart capability
    while True:
        # Car selection
        selected_car_index = car_selection_screen()
        if selected_car_index == -1:  # User chose to quit
            break
            
        # Name input
        player_name = name_input_screen(selected_car_index)
        if player_name is None:  # User chose to quit
            break
        
        # Initialize hand tracking with the updated USE_HANDS flag
        hands = HandIndexTracker() if USE_HANDS else None
        
        # Run game levels
        total_score = 0
        status, score1 = run(1, hands, high_score, player_name, selected_car_index)
        if status == "exit":  # User chose to exit
            break
        elif status == "restart":  # User crashed or restarted
            continue  # Go back to car selection
            
        total_score += score1
        
        status, score2 = run(2, hands, high_score, player_name, selected_car_index)
        if status == "exit":  # User chose to exit
            break
        elif status == "restart":  # User crashed or restarted
            continue  # Go back to car selection
            
        total_score += score2
        
        status, score3 = run(3, hands, high_score, player_name, selected_car_index)
        if status == "exit":  # User chose to exit
            break
        elif status == "restart":  # User crashed or restarted
            continue  # Go back to car selection
            
        total_score += score3
        
        # Save high score
        if total_score > high_score:
            with open("highscore.txt", "w") as f:
                f.write(str(total_score))
                
        # Show victory screen
        continue_game = victory_screen(total_score, high_score, hands, player_name)
        if not continue_game:  # User chose to quit
            break
            
        # Show end story screen (camera will be released here - CAMERA DISABLED)
        story_result = end_story_screen(hands)
        if not story_result:  # User chose to quit
            break
            
        # Show hand recognition screen (camera will be reinitialized here - CAMERA ENABLED)
        recognition_result = hand_recognition_screen()
        if not recognition_result:  # User chose to quit
            break
        # If recognition_result is True, user chose to restart, so continue the loop
    
    # Clean up - only release hands when completely exiting the game
    if 'hands' in locals() and hands:
        hands.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()