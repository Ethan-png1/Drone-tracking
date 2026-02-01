from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    

    results = model.train(

        data="../custom_data.yaml",

        # --- DURATION ---
        epochs=300,          
        patience=50,         
        
        # --- HARDWARE ---
        imgsz=640,
        batch=-1,            
        workers=8,
        device=0,            
        
        # --- HYPERPARAMETERS ---
        optimizer='auto',    
        cos_lr=True,         
        mixup=0.1,          
        
        # --- SYSTEM ---
        project="../Models/runs",
        name="yolov8_89k_run",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
