from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="../custom_data.yaml", 
        epochs=50, 
        imgsz=640,
        project="../Models/runs/train",
        name="drone_detector_v1"
    )

if __name__ =='__main__':
    main()


