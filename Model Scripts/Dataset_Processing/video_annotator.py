import cv2
import os

# --- CONFIGURATION ---
# Change this to the path of your video file
VIDEO_PATH = r"D:\SD\Ethan-dev\Datasets\videos\example.mp4"

# Folders where the extracted images and corresponding YOLO labels will be saved
OUTPUT_IMG_DIR = r"D:\SD\Ethan-dev\Datasets\custom_dataset\images"
OUTPUT_LBL_DIR = r"D:\SD\Ethan-dev\Datasets\custom_dataset\labels"

# How many frames to skip (e.g., 30 means we only look at 1 frame every second for a 30fps video)
# Set to 1 if you want to look at every single frame.
FRAME_INTERVAL = 30 

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# --- GLOBAL VARIABLES FOR DRAWING ---
drawing = False
ix, iy = -1, -1
boxes = []
clean_frame = None
display_frame = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, display_frame, clean_frame, boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_frame = clean_frame.copy()
            # Draw previously finalized boxes
            for box in boxes:
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # Draw the box currently being dragged
            cv2.rectangle(display_frame, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Only save if it's an actual box (greater than 5x5 pixels), not an accidental click
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def main():
    global clean_frame, display_frame, boxes
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return
        
    cv2.namedWindow("Video Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Annotator", 1280, 720)
    cv2.setMouseCallback("Video Annotator", draw_rectangle)
    
    frame_count = 0
    saved_count = 0
    
    print("\n--- Controls ---")
    print("Mouse   : Drag to draw bounding boxes")
    print("'s'     : Save current frame and labels, go to next frame")
    print("'n'     : Skip frame without saving")
    print("'c'     : Clear drawn boxes on current frame")
    print("'q'     : Quit annotator")
    print("----------------\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
            
        # Skip frames based on FRAME_INTERVAL
        if frame_count % FRAME_INTERVAL != 0:
            frame_count += 1
            continue
            
        boxes = []
        # Save clean copy without drawings to save to disk later
        clean_frame = frame.copy()
        # Initial display copy
        display_frame = frame.copy()
        
        while True:
            # Create a localized display copy for text overlay so it doesn't mess up drawings
            disp = display_frame.copy()
            cv2.putText(disp, f"Frame: {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(disp, "'s': Save, 'n': Skip, 'c': Clear, 'q': Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Video Annotator", disp)
            
            key = cv2.waitKey(20) & 0xFF
            
            # 1. QUIT
            if key == ord('q'):
                print("Quitting.")
                cap.release()
                cv2.destroyAllWindows()
                return 
                
            # 2. CLEAR DRAWING
            elif key == ord('c'):
                boxes = []
                display_frame = clean_frame.copy()
                
            # 3. SAVE
            elif key == ord('s'):
                if boxes:
                    img_name = f"frame_{frame_count:06d}.jpg"
                    lbl_name = f"frame_{frame_count:06d}.txt"
                    img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
                    lbl_path = os.path.join(OUTPUT_LBL_DIR, lbl_name)
                    
                    # Save the clean image without the rendered red boxes or text
                    cv2.imwrite(img_path, clean_frame) 
                    
                    h, w = clean_frame.shape[:2]
                    with open(lbl_path, "w") as f:
                        for box in boxes:
                            # YOLO format: <class> <x_center> <y_center> <width> <height>
                            x_center = ((box[0] + box[2]) / 2.0) / w
                            y_center = ((box[1] + box[3]) / 2.0) / h
                            b_width = (box[2] - box[0]) / float(w)
                            b_height = (box[3] - box[1]) / float(h)
                            
                            # Class ID is hardcoded to 0
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {b_width:.6f} {b_height:.6f}\n")
                    
                    saved_count += 1
                    print(f"Saved {img_name} and labels with {len(boxes)} box(es).")
                else:
                    print(f"Saved {frame_count} but there were *no* boxes! (Image skipped)")
                    # We skip saving if there's no bounding boxes drawn, or you could change it to save an empty txt.
                
                # Move to next frame
                break 
                
            # 4. SKIP
            elif key == ord('n') or key == 32: # 32 is spacebar
                print(f"Skipped frame {frame_count}")
                break 
                
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Extracted and labeled {saved_count} frames.")

if __name__ == '__main__':
    main()
