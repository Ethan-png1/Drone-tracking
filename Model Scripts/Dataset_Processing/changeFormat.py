from ultralytics import YOLO

def main():
    model = YOLO('../Models/best.pt')

    model.export(
        format='onnx', 
        half=True, 
        simplify=True,
        imgsz=640,
        opset=12,
        dynamic=False,
        data='custom_data.yaml'
    )
    

if __name__=='__main__':
    main()