from ultralytics.data.split import autosplit
from pathlib import Path

# Define the root directory of your dataset
DATASET_ROOT = Path("Datasets")
IMAGES_DIR = DATASET_ROOT / "images"

# Run the autosplit utility
# It will generate 'autosplit_train.txt' and 'autosplit_val.txt' 
# one directory level above the 'images' folder (i.e., inside 'custom_drone_dataset/')
autosplit(
    path=IMAGES_DIR,
    weights=(0.8, 0.2, 0.0),       # 80% Train, 20% Validation, 0% Test
    annotated_only=True            # Ensures only labeled images are included
)

print(f"Split files generated inside: {DATASET_ROOT.resolve()}")