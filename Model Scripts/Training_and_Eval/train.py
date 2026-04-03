from ultralytics import YOLO

def main():
    model = YOLO(r"D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V4\weights\best.pt")

    

    results = model.train(

        data=r"D:\SD\Ethan-dev\custom_data.yaml",

        # --- DURATION ---
        epochs=200,          
        patience=35,       
        
        # --- SAFETY ---
        # save_period=25,

        # --- HARDWARE ---
        imgsz=640,
        batch=-1,            
        workers=8,
        device=0,            
        
        # --- HYPERPARAMETERS ---
        optimizer='auto', 
        lr0=0.0001,   
        cos_lr=True,         
        box=8.0,      
        # --- Geometric ---
        degrees=90,
        translate=0.25,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.3,
        fliplr=0.3,
        mosaic=1.0,
        # --- ColorSpace ---
        hsv_h=0.015,  
        hsv_s=0.5,    
        hsv_v=0.4,    


        
        # --- SYSTEM ---
        project=r"D:\SD\Ethan-dev\Models\runs",
        name="yolov8_Drone_V6",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
