import cv2
import os
import shutil
import re

# --- CONFIGURATION ---
IMG_DIR = r"D:\SD\Ethan-dev\Datasets\BeefSet\train\images"
LBL_DIR = r"D:\SD\Ethan-dev\Datasets\BeefSet\train\labels"
DISCARD_IMG_DIR = r"D:\SD\Ethan-dev\Datasets\discards\images"
DISCARD_LBL_DIR = r"D:\SD\Ethan-dev\Datasets\discards\labels"
LAST_FILE_TRACKER = "last_reviewed_file.txt" 
BACKUP_YES_TRACKER = "backup_last_kept.txt"

os.makedirs(DISCARD_IMG_DIR, exist_ok=True)
os.makedirs(DISCARD_LBL_DIR, exist_ok=True)

# --- GLOBAL VARIABLES FOR DRAWING ---
drawing = False
ix, iy = -1, -1
new_boxes = []
clean_img = None
display_img = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, display_img, clean_img, new_boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_img = clean_img.copy()
            # Draw existing newly drawn boxes
            for box in new_boxes:
                cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # Draw the box currently being dragged
            cv2.rectangle(display_img, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Only save if it's an actual box, not an accidental click
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            new_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def draw_yolo_boxes(img, label_path, padding=10):
    h, w, _ = img.shape
    has_labels = False
    
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if len(parts) < 5: continue
                
                has_labels = True
                cls, x_c, y_c, nw, nh = map(float, parts)
                
                x1 = int((x_c - nw/2) * w)
                y1 = int((y_c - nh/2) * h)
                x2 = int((x_c + nw/2) * w)
                y2 = int((y_c + nh/2) * h)

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    if not has_labels:
        cv2.putText(img, "EMPTY LABEL - DRAW TO ADD", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

# --- SETUP ---
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
images.sort(key=natural_sort_key)
total = len(images)

start_index = 0
if os.path.exists(LAST_FILE_TRACKER):
    with open(LAST_FILE_TRACKER, "r") as f:
        last_name = f.read().strip()
    target_key = natural_sort_key(last_name)
    found = False
    for idx, name in enumerate(images):
        if natural_sort_key(name) > target_key:
            start_index = idx
            found = True
            break
    if found:
        print(f"Resuming at {images[start_index]}. (Last seen: {last_name})")

# --- MAIN LOOP ---
cv2.namedWindow("Review", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Review", 1280, 720)
cv2.setMouseCallback("Review", draw_rectangle)

i = start_index

while i < total:
    img_name = images[i]
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.exists(img_path):
        i += 1 
        continue

    raw_img = cv2.imread(img_path)
    if raw_img is None: 
        i += 1
        continue
    
    # Reset drawing variables for the new image
    new_boxes = []
    display_img = draw_yolo_boxes(raw_img.copy(), lbl_path, padding=15)
    cv2.rectangle(display_img, (0, 0), (1200, 60), (0, 0, 0), -1)
    cv2.putText(display_img, f"[{i+1}/{total}] {img_name}", (20, 40), 1, 1, (255, 255, 255), 2)
    
    clean_img = display_img.copy() # Base image used during mouse drag refresh

    # Real-time UI refresh loop (allows drawing to render instantly)
    key = -1
    while True:
        cv2.imshow("Review", display_img)
        key = cv2.waitKey(20) & 0xFF
        if key != 255: # Break out when any key is pressed
            break

    # 1. QUIT
    if key == ord('q'):
        print(f"Quitting. Next time you will start at: {img_name}")
        break
        
    # 2. CLEAR DRAWING
    elif key == ord('c'):
        continue # Just reloads the same image and resets boxes

    # 3. SAVE NEW DRAWING
    elif key == ord('s'):
        if new_boxes:
            h, w = raw_img.shape[:2]
            with open(lbl_path, 'a') as f: # Append mode, creates file if missing
                for box in new_boxes:
                    x_center = ((box[0] + box[2]) / 2.0) / w
                    y_center = ((box[1] + box[3]) / 2.0) / h
                    b_width = (box[2] - box[0]) / float(w)
                    b_height = (box[3] - box[1]) / float(h)
                    # Class ID is set to 0 by default
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {b_width:.6f} {b_height:.6f}\n")
            print(f"Saved {len(new_boxes)} new boxes to {lbl_path}")
            
        with open(BACKUP_YES_TRACKER, "w") as f: f.write(img_name)
        with open(LAST_FILE_TRACKER, "w") as f: f.write(img_name)
        i += 1

    # 4. GO BACK ('b')
    elif key == ord('b'):
        if i > 0:
            i -= 1
            while i > 0 and not os.path.exists(os.path.join(IMG_DIR, images[i])):
                i -= 1

    # 5. DISCARD ('n')
    elif key == ord('n'):
        shutil.move(img_path, os.path.join(DISCARD_IMG_DIR, img_name))
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(DISCARD_LBL_DIR, os.path.basename(lbl_path)))
        print(f"Discarded {img_name}")
        
        with open(LAST_FILE_TRACKER, "w") as f: f.write(img_name)
        i += 1

    # 6. KEEP / YES (Any other key, like spacebar)
    else:
        with open(BACKUP_YES_TRACKER, "w") as f: f.write(img_name)
        with open(LAST_FILE_TRACKER, "w") as f: f.write(img_name)
        i += 1

cv2.destroyAllWindows()