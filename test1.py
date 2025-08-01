import cv2
import numpy as np
import json
import os

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# hnee you can add  your colors and choose the priority(c'est optionnel)
DEFAULT_COLORS = {
    "Red": {"lower": [0, 120, 70], "upper": [10, 255, 255], "priority": 1},
    "Green": {"lower": [36, 50, 50], "upper": [89, 255, 255], "priority": 2},
    "Blue": {"lower": [90, 50, 50], "upper": [130, 255, 255], "priority": 3},
    "Yellow": {"lower": [15, 100, 100], "upper": [36, 255, 255], "priority": 4},
    "Orange": {"lower": [10, 100, 100], "upper": [20, 255, 255], "priority": 5},
    "Purple": {"lower": [130, 50, 50], "upper": [160, 255, 255], "priority": 6},
    "Pink": {"lower": [160, 50, 50], "upper": [180, 255, 255], "priority": 7},
    "Cyan": {"lower": [80, 50, 50], "upper": [100, 255, 255], "priority": 8},
    "Black": {"lower": [0, 0, 0], "upper": [180, 255, 40], "priority": 10}
}

def get_safe_db_path():
    """Get a reliable path for the JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "color_profiles.json")
    except:
        return "color_profiles.json"

COLOR_DB_FILE = get_safe_db_path()

def load_colors():
    """Load colors from JSON or return defaults"""
    try:
        if os.path.exists(COLOR_DB_FILE):
            with open(COLOR_DB_FILE, 'r') as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    return {**DEFAULT_COLORS, **saved}
    except:
        return DEFAULT_COLORS.copy()

def save_colors(colors):
    """Save colors to JSON"""
    try:
        with open(COLOR_DB_FILE, 'w') as f:
            json.dump({k: {"lower": v["lower"], "upper": v["upper"]} 
                     for k,v in colors.items() if k not in DEFAULT_COLORS}, f)
    except:
        print("Couldn't save colors")

color_ranges = load_colors()

# State variables
detection_active = False
current_color = ""
sampling = False
new_color_name = ""
last_stable_color = ""
stable_counter = 0

cv2.namedWindow("Color Detector")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    if detection_active:
        best_score = 0
        best_color = ""
        best_mask = None

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
        
        for color_name, values in color_ranges.items():
            lower = np.array(values["lower"], dtype=np.uint8)
            upper = np.array(values["upper"], dtype=np.uint8)
            mask = cv2.inRange(blurred, lower, upper)
            
            # Remove small noise
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            pixels = cv2.countNonZero(mask)
            
            score = pixels * (1 / values.get("priority", 5))
            
            if score > best_score and pixels > 1000:  # Higher threshold
                best_score = score
                best_color = color_name
                best_mask = mask

        # Stability check - only update if detected consistently
        if best_color == last_stable_color:
            stable_counter += 1
        else:
            stable_counter = 0
        
        if stable_counter > 5:  # Require 5 consistent frames(tnjm tbadel)
            current_color = best_color
        last_stable_color = best_color

        if best_mask is not None:
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        current_color = ""

    # UI
    cv2.putText(frame, f"Detected: {current_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    status = "ON" if detection_active else "OFF"
    color = (0, 255, 0) if detection_active else (0, 0, 255)
    cv2.putText(frame, f"Detection: {status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if sampling:
        cv2.putText(frame, "SAMPLING MODE", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Name: {new_color_name}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw crosshair and sample area
        cv2.rectangle(frame, (w//2-50, h//2-50), (w//2+50, h//2+50), (0, 255, 255), 2)
        cv2.line(frame, (w//2-15, h//2), (w//2+15, h//2), (0, 255, 255), 2)
        cv2.line(frame, (w//2, h//2-15), (w//2, h//2+15), (0, 255, 255), 2)
        
        # Show average color in sample area
        sample_area = hsv[h//2-50:h//2+50, w//2-50:w//2+50]
        if sample_area.size > 0:
            avg_hue = int(np.median(sample_area[:,:,0]))
            avg_sat = int(np.median(sample_area[:,:,1]))
            avg_val = int(np.median(sample_area[:,:,2]))
            cv2.putText(frame, f"HSV: {avg_hue},{avg_sat},{avg_val}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    cv2.putText(frame, "SPACE: Toggle | Q: Quit | S: Sample", 
                (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (255, 255, 255), 1)
    cv2.putText(frame, "ENTER: Save | ESC: Cancel", 
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (255, 255, 255), 1)

    cv2.imshow("Color Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        detection_active = not detection_active
        stable_counter = 0  # Reset stability counter when toggling
    elif key == ord('s'):
        sampling = True
        new_color_name = ""
    elif sampling:
        if key == 13:  # ENTER
            if new_color_name:
                sample_area = hsv[h//2-50:h//2+50, w//2-50:w//2+50]
                if sample_area.size > 0:
                    avg_hue = int(np.median(sample_area[:,:,0]))
                    # Wider range for sampled colors
                    color_ranges[new_color_name] = {
                        "lower": [max(0, avg_hue-15), 50, 50],
                        "upper": [min(179, avg_hue+15), 255, 255],
                        "priority": 5
                    }
                    save_colors(color_ranges)
            sampling = False
        elif key == 8:  # Backspace
            new_color_name = new_color_name[:-1]
        elif key == 27:  # ESC
            sampling = False
        elif 32 <= key <= 126:  # Printable ASCII
            new_color_name += chr(key)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()