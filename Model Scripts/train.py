from ultralytics import YOLO

def train(intial_model, model_name):
    model = YOLO(intial_model)

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
        ['yolo11n.pt', 'yolo11n_compare'],
        ['yolov10n.pt', 'yolov10n_compare']
    ]

    for model in model_matrix:
        initial_model_path = model[0]
        output_name = model[1]
        
        train(initial_model_path, output_name)
    

if __name__ =='__main__':
    main()
