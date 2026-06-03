from pathlib import Path
from ultralytics import YOLO
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_RUNS = _ROOT / "Models" / "runs"

def main():
    model = YOLO(str(_RUNS / "yolov8compare" / "best_v8n.pt"))

    print("--- Running Validation Metrics ---")
    metrics = model.val(
        data=str(_ROOT / "custom_data.yaml"),
        split='val',
        project=str(_RUNS / "yolov8compare" / "validation_results"),
        plots=True,
        save=True
    )

    print("Performance on Validation Set:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")

    print("\n--- Generating Report Images ---")

    # Update this path to your validation images directory
    val_images_path = str(_ROOT / "Datasets" / "Drone_dataset_smallobj" / "val" / "images")

    model.predict(
        source=val_images_path,
        project=str(_RUNS / "yolov8compare" / "report_images"),
        save=True,
        conf=0.25,
        line_width=2,
        max_det=10,
        save_txt=False,
        save_conf=True
    )
    print(f"Individual frames saved to: {_RUNS / 'yolov8compare' / 'report_images'}")

if __name__ =='__main__':
    main()