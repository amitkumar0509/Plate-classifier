# Indian License Plate Classification

This project is a YOLO-based image classification model for identifying Indian license plates into three categories:

- Standard
- Non Standard
- Invalid Plate

The model was trained to help separate well-formed plates from plates that do not follow the expected format. It is useful for ANPR pipelines, vehicle compliance screening, and downstream OCR workflows where plate quality matters.

## Project Summary

- **Task type:** Image classification
- **Model family:** Ultralytics YOLO classification
- **Trained weights:** [best.pt](runs/classify/plate_classifier-9/weights/best.pt)
- **Training script:** [train.py](train.py)
- **Training framework:** YOLO classification model from Ultralytics

## Classes

The model predicts one of these labels:

1. **Standard** - plates that follow the expected Indian license plate layout and appearance
2. **Non Standard** - plates that are readable but do not fully match the standard format
3. **Invalid Plate** - plates that are poorly formed, broken, unreadable, or unsuitable for recognition

## Dataset Used

The training set was prepared with the following approximate class distribution:

- **Standard:** 700 images
- **Non Standard:** 500 images
- **Invalid Plate:** 300 images

This distribution gives the model exposure to both common and edge-case plate types, improving robustness during real-world use.

## Model Output

The final and recommended trained model is stored at:

- [runs/classify/plate_classifier-9/weights/best.pt](runs/classify/plate_classifier-9/weights/best.pt)

Use `best.pt` for testing and inference. `last.pt` is mainly useful if you want to resume training from the final epoch.

## Key Features

- Fast and lightweight classification pipeline
- Built on a modern YOLO classification architecture
- Works well for pre-filtering plate quality before OCR
- Separate handling for standard, non-standard, and invalid plates
- Easy to retrain with new data or additional classes

## Advantages

- **Improves OCR accuracy:** invalid or badly formatted plates can be filtered before text extraction
- **Supports Indian plate styles:** designed around real-world Indian license plate variations
- **Better pipeline control:** helps decide whether to continue with detection/OCR or reject the image
- **Simple deployment:** the trained weights are compact and easy to integrate into scripts or apps
- **Scalable:** the same approach can be extended with more plate categories later

## Project Structure

```text
platetype val/
├── train.py
├── README.md
├── runs/
│   └── classify/
│       └── plate_classifier-9/
│           ├── args.yaml
│           ├── results.csv
│           ├── results.png
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           └── weights/
│               ├── best.pt
│               └── last.pt
```

## How to Train

Before training, point the script to your dataset folder if needed:

```bash
set PLATE_DATASET_DIR=C:\path\to\dataset
python train.py
```

The script uses `yolo11n-cls.pt` as the base model and saves outputs under `runs/classify/plate_classifier`.

## How to Test

Use the exported `best.pt` weights for evaluation or inference.

Example:

```python
from ultralytics import YOLO

model = YOLO("runs/classify/plate_classifier-9/weights/best.pt")
results = model.predict(source="test_image.jpg")
```

## Notes

- `best.pt` is the preferred model file for deployment and testing.
- `last.pt` is mainly for training continuation.
- If you add new training data, retrain the model and re-evaluate the validation results before replacing the current weights.
