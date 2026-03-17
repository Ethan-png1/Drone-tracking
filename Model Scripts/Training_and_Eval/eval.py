from ultralytics import YOLO

def main():
    model = YOLO('../Models/runs/yolov8_89k_run/weights/best.pt')

    metrics = model.val(
        data='../custom_data.yaml', 
        split='val',
        project='../Models/runs/yolov8_89k_run/validation_results',  # Main folder
    )

    print("Performance on Validation Set:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")
if __name__ =='__main__':
    main()