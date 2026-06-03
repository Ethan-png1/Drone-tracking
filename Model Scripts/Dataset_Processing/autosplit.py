from pathlib import Path
from ultralytics.data.split import autosplit

_ROOT = Path(__file__).resolve().parents[2]

autosplit(
    path=str(_ROOT / "Datasets" / "SimData" / "images"),
    weights=(1.0, 0, 0),
    annotated_only=True
)

autosplit(
    path=str(_ROOT / "Datasets" / "RealWorld" / "images"),
    weights=(0.8, 0.1, 0.1),
    annotated_only=True
)

