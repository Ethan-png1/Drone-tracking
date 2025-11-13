from ultralytics import YOLO

def train(intial_model, model_name):
    model = YOLO(intial_model)

    # use if your pre-loading a model
    model = model.load('yolov8n.pt')

    results = model.train(
        data="../custom_data.yaml", 
        epochs= 75, 
        imgsz=640,

        batch=-1,                  
        workers=8,                  
        optimizer='AdamW',

        project="../Models/runs",
        name=model_name
    )

def main():
    
    model_matrix = [
        ['yolov8n-p2.yaml','yolov8P2']
    ]

    for model in model_matrix:
        initial_model_path = model[0]
        output_name = model[1]
        
        train(initial_model_path, output_name)
    

if __name__ =='__main__':
    main()
