from ultralytics import YOLO

def main():
    model = YOLO("yolov9t.pt")

    results = model.train(
        data="../custom_data.yaml", 
        epochs=50, 
        imgsz=640,
        project="../Models/runs",
        name="drone_detector_v3"
    )

if __name__ =='__main__':
    main()


