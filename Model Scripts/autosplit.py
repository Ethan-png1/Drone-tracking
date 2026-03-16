from ultralytics.data.split import autosplit
from pathlib import Path

# Define the root directory of your dataset
DATASET_ROOT = Path(r"D:\SD\Ethan-dev\Datasets\Droneset")
IMAGES_DIR = DATASET_ROOT / "images"

# Run the autosplit utility
# It will generate 'autosplit_train.txt' and 'autosplit_val.txt' 
# one directory level above the 'images' folder (i.e., inside 'custom_drone_dataset/')
autosplit(
    path=IMAGES_DIR,
    weights=(0.7, 0.2, 0.1),       # 80% Train, 20% Validation, 0% Test
    annotated_only=False            # Ensures only labeled images are included
)

print(f"Split files generated inside: {DATASET_ROOT.resolve()}")