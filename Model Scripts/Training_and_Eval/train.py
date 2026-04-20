from ultralytics import YOLO
from ultralytics.utils import LOGGER

def main():
    model = YOLO(r"D:\SD\Ethan-dev\Models\runs\yolov8_Drone_V7\weights\best.pt")

    # Augmentation decay schedule
    # Values decay linearly from START -> END over the course of training
    DECAY_SCHEDULE = {
        "degrees":     (90,    30),
        "translate":   (0.25,  0.05),
        "scale":       (0.5,   0.2),
        "shear":       (5.0,   1.0),
        "perspective": (0.001, 0.0002),
        "flipud":      (0.15,  0.05),   # Already lowered from 0.3
        "fliplr":      (0.15,  0.05),   # Already lowered from 0.3
        "mosaic":      (1.0,   0.5),
    }

    def decay_augmentation(trainer):
        epoch      = trainer.epoch
        total      = trainer.epochs
        progress   = epoch / max(total - 1, 1)  # 0.0 -> 1.0

        for key, (start, end) in DECAY_SCHEDULE.items():
            current = start + (end - start) * progress
            trainer.args.__dict__[key] = current

        if epoch % 10 == 0:  # Log every 10 epochs to avoid spam
            sample = {k: round(v + (e - v) * progress, 5)
                      for k, (v, e) in DECAY_SCHEDULE.items()}
            LOGGER.info(f"[Epoch {epoch + 1}/{total}] Augmentation state: {sample}")

    model.add_callback("on_train_epoch_start", decay_augmentation)

    results = model.train(
        data=r"D:\SD\Ethan-dev\custom_data.yaml",

        # --- DURATION ---
        epochs=200,
        patience=30,

        # --- HARDWARE ---
        imgsz=640,
        batch=-1,
        workers=8,
        device=0,

        # --- HYPERPARAMETERS ---
        optimizer='auto',
        lr0=0.0001,
        cos_lr=True,

        # --- Geometric (initial values, decay_augmentation will override each epoch) ---
        degrees=90,
        translate=0.25,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.15,        # Lowered from 0.3
        fliplr=0.15,        # Lowered from 0.3
        mosaic=1.0,
        close_mosaic=45,    # Increased from 20 — longer stable phase at end

        # --- ColorSpace ---
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        # --- SYSTEM ---
        project=r"D:\SD\Ethan-dev\Models\runs",
        name="yolov8_Drone_V9",

        # --- MEMORY SAFETY ---
        cache=False,
    )


if __name__ == '__main__':
    main()
