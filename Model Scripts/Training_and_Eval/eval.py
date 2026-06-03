from pathlib import Path
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "Models" / "runs"

MODELS = [
    # _RUNS / "yolov8_Drone_V3" / "weights" / "best.pt",
    # _RUNS / "yolov8_Drone_V4" / "weights" / "best.pt",
    # _RUNS / "yolov8_Drone_V5" / "weights" / "best.pt",
    _RUNS / "yolov8_Drone_V6" / "weights" / "best.pt",
    # _RUNS / "yolov8_Drone_V7" / "weights" / "best.pt",
]

def main():
    for model_path in MODELS:
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(_ROOT / "custom_data.yaml"),
            split='val',
            save=False,
            plots=False,
            verbose=False,
        )
        print(f"\n{model_path.parent.parent.name}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50:    {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
