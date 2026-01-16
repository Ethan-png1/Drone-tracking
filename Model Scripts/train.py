from ultralytics import YOLO

def train(model_weights, output_name):
    model = YOLO(model_weights)

    freeze_layers = 10 if ('yolov8l' in model_weights or 'yolov8x' in model_weights) else 0

    results = model.train(
        data="../custom_data.yaml", 
        epochs= 75, 
        imgsz=640,
        batch=-1,                  
        workers=8,                  
        optimizer='AdamW',
        project="../Models/runs",
        name=output_name,

        patience=10,

        freeze=freeze_layers
    )

def main():
    
    model_matrix = [
        ['yolov8n.pt', 'yolov8_Nano'],
        ['yolov8s.pt', 'yolov8_Small'],
        ['yolov8m.pt', 'yolov8_Medium'],
        ['yolov8l.pt', 'yolov8_Large'],
        ['yolov8x.pt', 'yolov8_XLarge']
    ]

    for entry in model_matrix:
        weights = entry[0]
        name = entry[1]
        
       
        train(weights, name)
    

if __name__ =='__main__':
    main()
