from ultralytics import YOLO

MODELS = [
    #'../../Models/runs/yolov8_Drone_V3/weights/best.pt',
    #'../../Models/runs/yolov8_Drone_V4/weights/best.pt',
    #'../../Models/runs/yolov8_Drone_V5/weights/best.pt',
    '../../Models/runs/yolov8_Drone_V6/weights/best.pt',
    #'../../Models/runs/yolov8_Drone_V7/weights/best.pt',
]

def main():
    for model_path in MODELS:
        model = YOLO(model_path)
        metrics = model.val(
            data='../../custom_data.yaml',
            split='val',
            save=False,
            plots=False,
            verbose=False,
        )
        print(f"\n{model_path}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50:    {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
