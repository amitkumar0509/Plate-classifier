import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

from ultralytics import YOLO

def main():
    model = YOLO("yolo11n-cls.pt")
    dataset_dir = os.environ.get("PLATE_DATASET_DIR", "dataset")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. Set PLATE_DATASET_DIR to your local dataset path."
        )

    results = model.train(
        data=dataset_dir,
        epochs=100,
        imgsz=224,
        batch=8,           # <-- Lowered to 8 to comfortably fit in 6GB VRAM
        device=0,            
        project="runs/classify",
        name="plate_classifier",
        workers=0,         
        cache=False,
        amp=False           # Keep AMP true first; if it still drops, you can toggle to False
    )

if __name__ == "__main__":
    main()