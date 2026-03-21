from ultralytics import YOLO

def main():
    model = YOLO(r"D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V4\weights\last.pt")

    

    results = model.train(

        data=r"D:\SD\Ethan-dev\custom_data.yaml",

        # --- DURATION ---
        epochs=300,          
        patience=50,         
        
        # --- SAFETY ---
        # save_period=25,

        # --- HARDWARE ---
        imgsz=640,
        batch=16,            
        workers=8,
        device=0,            
        
        # --- HYPERPARAMETERS ---
        optimizer='auto', 
        lr0=0.001,   
        cos_lr=True,         
        mosaic=1.0,
        copy_paste=0.0,          
        
        # --- SYSTEM ---
        project=r"D:\SD\Ethan-dev\Models\runs",
        name="yolov8_Drone_V4",  
        
        # --- MEMORY SAFETY ---
        cache=False,                     
    )


if __name__ =='__main__':
    main()
