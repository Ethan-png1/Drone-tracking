from ultralytics import YOLO

def main():
    model = YOLO('../Models/runs/train/drone_detector_v1/weights/best.pt')

    metrics = model.val(
        data='../custom_data.yaml', 
        split='val',
        project='../Models/runs/train/drone_detector_v1/validation_results',  # Main folder
        name='first_run'               # Subfolder for this specific run
    )

    print("Performance on Validation Set:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")
if __name__ =='__main__':
    main()