import cv2
import os
from ultralytics import YOLO

def main():
    # 1. Load your best model
    model = YOLO(r'D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V5\weights\best.pt')

    # 2. CONFIGURATION
    # REPLACE THIS with the actual path to your video file
    video_path = r"D:\SD\Ethan-dev\Datasets\videos\testsim.mp4" 
    
    # How often to process a frame? (Set this to 1 for smooth video playback)
    frame_interval = 1 

    # 3. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    frame_count = 0
    print(f"Playing video... rendering every {frame_interval}th frame. Press 'q' to quit.")

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

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone! Video playback finished.")

if __name__ == '__main__':
    main()