import cv2
import os
import time
import numpy as np

# ================= CONFIGURATION =================
# Replace with your ESP32 IP
ESP32_IP = "192.168.1.26" 
# Note: standard CameraWebServer uses port 81 for stream
URL = f"http://{ESP32_IP}:81/stream"

# Settings
IMG_SIZE = 96
DATA_DIR = "dataset"
CLASSES = ["rock", "paper", "scissors"]
# =================================================

# Create directories
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

print(f"Connecting to {URL}...")
cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("Error: Could not open video stream. Check IP or Wi-Fi.")
    exit()

print("-------------------------------------------------")
print("Press 'r' to save ROCK")
print("Press 'p' to save PAPER")
print("Press 's' to save SCISSORS")
print("Press 'q' to QUIT")
print("-------------------------------------------------")

count = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in CLASSES}

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        time.sleep(0.5)
        continue

    # 1. CENTER CROP (To make it square)
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img_cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # 2. RESIZE to 96x96
    img_resized = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))

    # 3. GRAYSCALE
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Display (Zoomed in x4 so you can see better)
    display_img = cv2.resize(img_gray, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Overlay text
    status_text = f"R:{count['rock']} P:{count['paper']} S:{count['scissors']}"
    cv2.putText(display_img, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Data Collector", display_img)

    key = cv2.waitKey(1) & 0xFF
    save_label = None

    if key == ord('r'): save_label = "rock"
    elif key == ord('p'): save_label = "paper"
    elif key == ord('s'): save_label = "scissors"
    elif key == ord('q'): break

    if save_label:
        timestamp = int(time.time() * 1000)
        filename = os.path.join(DATA_DIR, save_label, f"{timestamp}.jpg")
        cv2.imwrite(filename, img_gray)
        count[save_label] += 1
        print(f"Saved {save_label}. Total: {count[save_label]}")

cap.release()
cv2.destroyAllWindows()