from ultralytics import YOLO
import os
import yaml

def main():
    # 1. Load the model
    model_path = '../Models/runs/yolov8compare/best_v8n.pt'
    model = YOLO(model_path)

    # 2. RUN VALIDATION (For the Numbers)
    # plots=True saves the confusion matrix and the batch mosaics
    print("--- Running Validation Metrics ---")
    metrics = model.val(
        data='../custom_data.yaml', 
        split='val',
        project='../Models/runs/yolov8compare/validation_results',
        plots=True, # This saves the standard validation plots
        save=True   
    )

    print("Performance on Validation Set:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")

    # 3. RUN PREDICTION (For the Report Images)
    # This generates individual high-res images, not grids
    print("\n--- Generating Report Images ---")
    
    # Load the data.yaml to find where your validation images are
    with open('../custom_data.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    # Assuming your yaml has 'val: path/to/images'
    val_images_path = r"C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\Drone_dataset_smallobj\val\images" 
    
    # Run inference specifically to save images
    model.predict(
        source=val_images_path,
        project='../Models/runs/yolov8compare/report_images', # Separate folder
        save=True,
        conf=0.25,      # Filter out low confidence noise for cleaner images
        line_width=2,   # Make boxes slightly thinner so they don't cover the drone
        max_det=10,     # Limit detections per frame (cleaner)
        save_txt=False, # We just want images
        save_conf=True  # Show confidence scores
    )
    print(f"Individual frames saved to: ../Models/runs/yolov8compare/report_images")

if __name__ =='__main__':
    main()