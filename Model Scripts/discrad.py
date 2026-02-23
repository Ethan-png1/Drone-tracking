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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 1. Get and SORT images
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
images.sort(key=natural_sort_key)
total = len(images)

# 2. Load progress (CRASH-FREE VERSION)
start_index = 0
if os.path.exists(LAST_FILE_TRACKER):
    with open(LAST_FILE_TRACKER, "r") as f:
        last_name = f.read().strip()
    
    # Pre-calculate the natural sort key for our target string
    target_key = natural_sort_key(last_name)
    
    found = False
    for idx, name in enumerate(images):
        # Compare them using natural sorting math, not alphabetical string math
        if natural_sort_key(name) > target_key:
            start_index = idx
            found = True
            break
    
    if found:
        print(f"Resuming at {images[start_index]}. (Last seen: {last_name})")
    else:
        print(f"Note: Couldn't find next file after {last_name}.")

def draw_yolo_boxes(img, label_path, padding=10):
    h, w, _ = img.shape
    if not os.path.exists(label_path):
        cv2.putText(img, "NO LABEL", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if len(parts) < 5: continue
            
            cls, x, y, nw, nh = map(float, parts)
            
            x1 = int((x - nw/2) * w)
            y1 = int((y - nh/2) * h)
            x2 = int((x + nw/2) * w)
            y2 = int((y + nh/2) * h)

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

# --- MAIN LOOP ---
cv2.namedWindow("Review", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Review", 1280, 720)

for i in range(start_index, total):
    img_name = images[i]
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    raw_img = cv2.imread(img_path)
    if raw_img is None: continue
    
    display_img = draw_yolo_boxes(raw_img.copy(), lbl_path, padding=15)
    cv2.putText(display_img, f"[{i+1}/{total}] {img_name}", (20, 40), 1, 1, (255, 255, 255), 2)

    cv2.imshow("Review", display_img)
    key = cv2.waitKey(0) & 0xFF

    # QUIT
    if key == ord('q'):
        print(f"Quitting. Next time you will start at: {img_name}")
        break

    # DISCARD
    if key == ord('n'):
        shutil.move(img_path, os.path.join(DISCARD_IMG_DIR, img_name))
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(DISCARD_LBL_DIR, os.path.basename(lbl_path)))
        print(f"Discarded {img_name}")
    else:
        # KEEP / YES (Any key that isn't 'q' or 'n')
        # Save this to the backup tracker because it was kept
        with open(BACKUP_YES_TRACKER, "w") as f:
            f.write(img_name)

    # ALWAYS save the current image name as the "last seen" before moving to the next
    with open(LAST_FILE_TRACKER, "w") as f:
        f.write(img_name)

cv2.destroyAllWindows()