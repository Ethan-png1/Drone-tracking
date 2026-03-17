import cv2
import os
from ultralytics import YOLO

def main():
    # 1. Load your best model
    model = YOLO('../Models/runs/yolov8_89k_run/weights/best.pt')

    # 2. CONFIGURATION
    # REPLACE THIS with the actual path to your video file
    video_path = r"D:\SD\Ethan-dev\Datasets\videos\video.mp4" 
    
    # Where to save the images
    output_folder = '../Models/runs/yolov8_89k_run/video_results'
    
    # How often to save a frame? (30 = save 1 frame per second if video is 30fps)
    # Set this to 1 if you want EVERY single frame (Warning: creates thousands of files)
    frame_interval = 30 

    # 3. Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # 4. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    frame_count = 0
    saved_count = 0
    print(f"Processing video... saving every {frame_interval}th frame.")

    while True:
        success, frame = cap.read()
        if not success:
            break # End of video

        # Only process specific frames to save space/time
        if frame_count % frame_interval == 0:
            # Run YOLO inference on this single frame
            results = model.predict(
                frame, 
                conf=0.25, 
                save=False,     # Don't let YOLO save; we will do it manually
                verbose=False   # Keep terminal quiet
            )

            # Plot the boxes on the frame
            # line_width=2 keeps boxes thin so they don't cover the drone
            annotated_frame = results[0].plot(line_width=2)

            # Save the image to your folder
            filename = f"frame_{frame_count:05d}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, annotated_frame)
            
            saved_count += 1
            print(f"Saved: {filename}")

        frame_count += 1

    cap.release()
    print(f"\nDone! Saved {saved_count} images to:\n{output_folder}")

if __name__ == '__main__':
    main()