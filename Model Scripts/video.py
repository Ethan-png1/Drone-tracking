from ultralytics import YOLO

def main():
    
    model = YOLO('../Models/runs/GigaDrone1/weights/best.pt')

    results = model.track("../Datasets/videos/cam0.mp4",save=True,tracker='bytetrack.yaml',project='../Models/runs/GigaDrone1')
    
if __name__=='__main__':
    main()