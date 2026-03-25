from ultralytics import YOLO

def main():
    model = YOLO(r"D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V4\weights\best.pt")

    

    results = model.train(

        data=r"D:\SD\Ethan-dev\custom_data.yaml",

        # --- DURATION ---
        epochs=100,          
        patience=50,
        freeze=10,         
        
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
        mosaic=1.0,
        copy_paste=0.0,
        mixup=0.1,                    
        scale=0.5,
        hsv_h=0.015,  # minor hue shifts
        hsv_s=0.7,    # helps model ignore "perfect" sim colors
        hsv_v=0.4,    # helps model handle real-world lighting          
        
        # --- SYSTEM ---
        project=r"D:\SD\Ethan-dev\Models\runs",
        name="yolov8_Drone_V5",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
