from ultralytics import YOLO

def main():
    model = YOLO("best.pt")

    

    results = model.train(

        data="../custom_data.yaml",

        # --- DURATION ---
        epochs=500,          
        patience=50,         
        
        # --- SAFETY ---
        save_period=25,

        # --- HARDWARE ---
        imgsz=640,
        batch=-1,            
        workers=8,
        device=0,            
        
        # --- HYPERPARAMETERS ---
        optimizer='auto', 
        lr0=0.001,   
        cos_lr=True,         
        mosaic=1.0,
        copy_paste=0.2,          
        
        # --- SYSTEM ---
        project="../Models/runs",
        name="yolov8_Drone_V3",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
