from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    

    results = model.train(

        data="../custom_data.yaml",

        # --- DURATION ---
        epochs=150,          
        patience=15,         
        
        # --- HARDWARE ---
        imgsz=640,
        batch=-1,            
        workers=8,
        device=0,            
        
        # --- HYPERPARAMETERS ---
        optimizer='auto',    
        cos_lr=True,         
        mixup=0.0,          
        
        # --- SYSTEM ---
        project="../Models/runs",
        name="yolov8_Drone_V2",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
