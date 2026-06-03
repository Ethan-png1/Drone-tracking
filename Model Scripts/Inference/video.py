import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO


def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def pick_video_file():
    return pick_file(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"), ("All files", "*.*")]
    )


class DroneKalmanFilter:
    """
    Tracks a single drone's position and velocity using a Kalman filter.
    State vector: [x, y, vx, vy]
      x, y   = centre of bounding box
      vx, vy = velocity in pixels/frame
    """
    def __init__(self, initial_box):
        self.kf = cv2.KalmanFilter(4, 2)   # 4 state vars, 2 measured vars

        # Transition matrix — how state evolves each frame
        # x  = x  + vx
        # y  = y  + vy
        # vx = vx       (constant velocity model)
        # vy = vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix — we only observe x, y (not velocity)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise — how much we trust the physics model
        # Higher = filter adapts faster to direction changes
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise — how much we trust the detector
        # Higher = filter smooths out jittery detections
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initialise state from first detection
        cx, cy = self._box_centre(initial_box)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

        self.last_box_size = self._box_size(initial_box)
        self.missing_frames = 0
        self.history = []          # list of (cx, cy) for drawing trail

    def _box_centre(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _box_size(self, box):
        x1, y1, x2, y2 = box
        return abs(x2 - x1), abs(y2 - y1)

    def predict(self):
        """Step the physics model forward one frame."""
        predicted = self.kf.predict()
        cx, cy = float(predicted[0]), float(predicted[1])
        self.history.append((cx, cy))
        if len(self.history) > 40:      # keep last 40 frames of trail
            self.history.pop(0)
        return cx, cy

    def update(self, box):
        """Feed a real detection into the filter."""
        cx, cy = self._box_centre(box)
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(measurement)
        self.last_box_size = self._box_size(box)
        self.missing_frames = 0

    def get_search_zone(self, missing_frames, base_margin=40, growth=15):
        """
        Returns (cx, cy, half_w, half_h) of the search zone.
        The zone grows each frame the drone is missing — reflecting
        increasing position uncertainty as physics predictions drift.
        """
        state = self.kf.statePost
        cx, cy = float(state[0]), float(state[1])
        vx, vy = float(state[2]), float(state[3])

        # Project forward: where will the drone be in `missing_frames` more frames?
        proj_cx = cx + vx * missing_frames
        proj_cy = cy + vy * missing_frames

        bw, bh = self.last_box_size
        margin = base_margin + growth * missing_frames   # zone expands over time
        half_w = bw / 2 + margin
        half_h = bh / 2 + margin

        return proj_cx, proj_cy, half_w, half_h

    def get_velocity(self):
        vx = float(self.kf.statePost[2])
        vy = float(self.kf.statePost[3])
        return vx, vy


class HysteresisTracker:
    def __init__(self, conf_init=0.5, conf_keep=0.2, conf_floor=0.05, max_missing=15):
        """
        conf_init  : confidence needed to START a new track
        conf_keep  : confidence needed to keep track when actively detected
        conf_floor : lowest the threshold will ever drop to during search
        max_missing: frames before track is killed
        """
        self.conf_init      = conf_init
        self.conf_keep      = conf_keep
        self.conf_floor     = conf_floor
        self.max_missing    = max_missing
        self.confirmed_ids  = set()
        self.active_ids     = {}
        self.kalman_filters = {}

    def get_dynamic_threshold(self, missing_frames):
        """
        Linearly interpolates from conf_keep down to conf_floor
        as missing_frames goes from 0 to max_missing.

        Frame 0  (just lost): threshold = conf_keep (e.g. 0.20)
        Frame 7  (halfway):   threshold = ~0.125
        Frame 15 (about to die): threshold = conf_floor (e.g. 0.05)
        """
        ratio = missing_frames / self.max_missing
        return self.conf_keep - (self.conf_keep - self.conf_floor) * ratio

    def update(self, boxes, track_ids, confs):
        seen = set()
        output = []

        for box, tid, conf in zip(boxes, track_ids, confs):
            tid = int(tid)
            seen.add(tid)

            if tid not in self.confirmed_ids:
                # Brand new track — needs full conf_init
                if conf >= self.conf_init:
                    self.confirmed_ids.add(tid)
                    self.active_ids[tid] = 0
                    self.kalman_filters[tid] = DroneKalmanFilter(box)
                    output.append((box, tid, conf, False))

            else:
                kf = self.kalman_filters[tid]
                kf.predict()

                missing = self.active_ids[tid]
                dynamic_threshold = self.get_dynamic_threshold(missing)

                if conf >= dynamic_threshold:
                    # Re-detection passed the (possibly lowered) threshold
                    kf.update(box)
                    self.active_ids[tid] = 0    # reset — drone found again
                    output.append((box, tid, conf, False))
                else:
                    self.active_ids[tid] += 1

        # Tracks not seen at all this frame
        for tid in list(self.active_ids.keys()):
            if tid not in seen and tid in self.confirmed_ids:
                kf = self.kalman_filters[tid]
                kf.missing_frames += 1
                kf.predict()
                self.active_ids[tid] += 1

                if self.active_ids[tid] <= self.max_missing:
                    missing = self.active_ids[tid]
                    dynamic_threshold = self.get_dynamic_threshold(missing)
                    output.append((None, tid, dynamic_threshold, True))

        # Kill dead tracks
        dead = [tid for tid, m in self.active_ids.items() if m > self.max_missing]
        for tid in dead:
            del self.active_ids[tid]
            del self.kalman_filters[tid]
            self.confirmed_ids.discard(tid)

        return output


def draw_arrow(frame, cx, cy, vx, vy, color=(0, 200, 255), scale=5):
    """Draw a velocity arrow from the drone centre."""
    start = (int(cx), int(cy))
    end   = (int(cx + vx * scale), int(cy + vy * scale))
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)


def draw_search_zone(frame, cx, cy, half_w, half_h, missing_frames, max_missing, vx, vy, dynamic_threshold):
    x1 = int(cx - half_w)
    y1 = int(cy - half_h)
    x2 = int(cx + half_w)
    y2 = int(cy + half_h)

    # Colour fades orange -> red as threshold drops and time runs out
    decay_ratio = 1.0 - (missing_frames / max_missing)
    color = (0, int(165 * decay_ratio), 255)

    dash, gap = 10, 6
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 1)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 1)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 1)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 1)

    draw_arrow(frame, cx, cy, vx, vy, color=color)

    frames_left = max_missing - missing_frames
    label = f"Need conf > {dynamic_threshold:.2f}  ({frames_left}f left)"
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def draw_trail(frame, history, color=(0, 255, 180)):
    """Draw a fading motion trail behind the drone."""
    for i in range(1, len(history)):
        alpha = i / len(history)
        thickness = max(1, int(alpha * 3))
        faded = tuple(int(c * alpha) for c in color)
        pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
        pt2 = (int(history[i][0]),     int(history[i][1]))
        cv2.line(frame, pt1, pt2, faded, thickness)


def main():
    print("Please select a model file...")
    model_path = pick_file(
        title="Select a Model File",
        filetypes=[("Model files", "*.pt *.onnx"), ("All files", "*.*")]
    )
    if not model_path:
        print("No model selected. Exiting.")
        return
    model = YOLO(model_path)

    print("Please select a video file...")
    video_path = pick_video_file()
    if not video_path:
        print("No file selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv2.namedWindow("YOLOv8 Drone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Drone Tracker", min(vid_w, 1280), min(vid_h, 720))

    skip_frames = 2  # process every Nth frame (1 = no skipping, 2 = 2x speed, etc.)

    base, _ = os.path.splitext(video_path)
    out_path = base + "_tracked.mp4"
    out_fps = src_fps / skip_frames
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (vid_w, vid_h))
    print(f"Saving output to: {out_path}")

    tracker = HysteresisTracker(
        conf_init=0.5,
        conf_keep=0.2,
        max_missing=15
    )
    frame_count = 0

    print("Running... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        results = model.track(
            frame,
            tracker="botsort.yaml",
            conf=0.03,
            iou=0.45,
            imgsz=640,
            persist=True,
            save=False,
            verbose=False
        )

        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes  = results[0].boxes.xyxy.cpu().numpy()
            ids    = results[0].boxes.id.cpu().numpy()
            confs  = results[0].boxes.conf.cpu().numpy()
        else:
            boxes, ids, confs = [], [], []

        approved = tracker.update(boxes, ids, confs)

        for box, tid, conf, is_missing in approved:
            kf = tracker.kalman_filters[tid]

            if not is_missing:
                # Active detection — draw normal box + trail
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                vx, vy = kf.get_velocity()

                draw_trail(frame, kf.history)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_arrow(annotated_frame, cx, cy, vx, vy)

                label = f"Drone {tid}  {conf:.2f}  v=({vx:.1f},{vy:.1f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                missing = tracker.active_ids[tid]
                proj_cx, proj_cy, half_w, half_h = kf.get_search_zone(missing)
                vx, vy = kf.get_velocity()

                draw_trail(frame, kf.history)
                draw_search_zone(
                    annotated_frame,
                    proj_cx, proj_cy,
                    half_w, half_h,
                    missing,
                    tracker.max_missing,   # pass max so we can compute frames_left
                    vx, vy,
                    conf                   # this is already the decayed_conf from tracker.update()
                )

        writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Drone Tracker", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()