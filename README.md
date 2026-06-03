# Drone Detection in High-Clutter Backgrounds

A YOLOv8-based pipeline for detecting and tracking small drones in visually complex environments. Uses Unreal Engine 4 to generate synthetic training data alongside real-world footage, and runs real-time inference with a Kalman filter tracker.

## Project Structure

```
├── Model Scripts/
│   ├── Dataset_Processing/   # Tools for building and cleaning datasets
│   ├── Inference/            # Run detection on images or video
│   └── Training_and_Eval/    # Train and evaluate models
├── custom_data.yaml          # YOLO dataset config (points to ./Datasets)
└── requirements.txt          # Python dependencies
```

> **Note:** `Datasets/`, `Models/`, and the `Blocks/` UE4 project are excluded from this repository due to size. Model weights (`.pt`, `.onnx`) must be downloaded or trained locally.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repo-folder>
```

### 2. Set up Python environment

This project requires **Python 3.11.9**. Using a different version may cause dependency conflicts.

```bash
python -m venv venv

# Windows
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch with CUDA (do this first)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run inference on a video

Prompts you to select a model file and a video file via file dialogs.

```bash
python "Model Scripts/Inference/video.py"
```

**Controls during playback:**
- `q` — quit

### Train a model

Edit `BASE_MODEL` and `RUN_NAME` at the top of the script, then:

```bash
python "Model Scripts/Training_and_Eval/train.py"
```

By default this fine-tunes from `yolov8m.pt` (auto-downloaded by ultralytics). Point `BASE_MODEL` to a previous checkpoint to continue training.

### Evaluate a model

Edit the `MODELS` list in the script to include your trained checkpoint paths, then:

```bash
python "Model Scripts/Training_and_Eval/eval.py"
```

### Dataset tools

| Script | Purpose |
|--------|---------|
| `Dataset_Processing/video_annotator.py` | Extract and annotate frames from a video |
| `Dataset_Processing/discard.py` | Interactively review and discard bad samples |
| `Dataset_Processing/dataset_merger.py` | Merge multiple datasets with class ID remapping |
| `Dataset_Processing/autosplit.py` | Split datasets into train/val/test splits |
| `Dataset_Processing/changeFormat.py` | Export a trained model to ONNX format |

## Dataset Layout

Place your datasets under `Datasets/` following this structure:

```
Datasets/
├── SimData/
│   └── images/          # Synthetic frames from UE4
├── RealWorld/
│   ├── images/
│   └── labels/          # YOLO-format .txt annotations
└── ...
```

After placing data, run `autosplit.py` to generate the split list files referenced by `custom_data.yaml`.

## Key Dependencies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch (CUDA 12.1)
- OpenCV
- NumPy / SciPy


## Demo
https://github.com/user-attachments/assets/56f75f33-b73d-4661-920e-65c260f5bd21

