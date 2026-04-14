import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def pick_video_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return video_path

def main():
    # 1. Load your best model
    model = YOLO(r'D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V6\weights\best.pt')

    # 2. Pick video
    print("Please select a video file...")
    video_path = pick_video_file()

    if not video_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected: {video_path}")

    # 3. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    frame_count = 0
    frame_interval = 1

    # Tracker options: "bytetrack.yaml" or "botsort.yaml"
    TRACKER = "botsort.yaml"

    print(f"Running with {TRACKER}... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            # 4. Use .track() instead of .predict()
            results = model.track(
                frame,
                tracker=TRACKER,    # swap to "botsort.yaml" anytime
                conf=0.25,
                iou=0.5,            # overlap threshold for matching boxes
                persist=True,       # keeps track IDs stable across frames
                save=False,
                verbose=False
            )

            # 5. Plot — track IDs are drawn automatically
            annotated_frame = results[0].plot(line_width=2)

            # 6. Optionally print track info to terminal
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(track_id)
                    print(f"  Frame {frame_count} | Drone ID {tid} @ [{x1},{y1},{x2},{y2}]")

            cv2.imshow("YOLOv8 Drone Tracker", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == '__main__':
    main()