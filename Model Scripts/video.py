from ultralytics import YOLO

def main():
    
    model = YOLO('../Models/runs/train/drone_detector_v1/weights/best.pt')

    results = model.track("../Datasets/videos/cam0.mp4",save=True,tracker='bytetrack.yaml',project='../Models/runs/train/drone_detector_v1')
    
if __name__=='__main__':
    main()