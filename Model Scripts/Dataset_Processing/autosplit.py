from ultralytics.data.split import autosplit



autosplit(
    path=r"D:\SD\Ethan-dev\Datasets\SimData\images",
    weights=(1.0, 0, 0),       
    annotated_only=True            
)

autosplit(
    path=r"D:\SD\Ethan-dev\Datasets\RealWorld\images",
    weights=(0.8, 0.1, 0.1),       
    annotated_only=True            
)

