# Contributing to ML Studio

Thanks for helping! ML Studio is free and open source (AGPL-3.0).

## Run it locally

```
pip install -r requirements.txt
python run.py          # http://127.0.0.1:8000
```

Optional: `torch torchvision`, `ultralytics`, `xgboost lightgbm catboost` unlock more models.
Click **Load sample datasets** on the home page for quick test data.

## Code layout

| Path | What it does |
|---|---|
| `app/server.py` | FastAPI REST API |
| `app/storage.py` | file-based projects, items and runs (`workspace/`) |
| `app/trainers.py` | scikit-learn, PyTorch and YOLO trainers, prediction, model catalog |
| `app/features.py` | image features (HOG, color histogram) and classifier/regressor catalogs |
| `app/samples.py` | built-in synthetic sample datasets |
| `app/static/index.html` | the whole UI (vanilla JS, no build step) |

## Ideas

* More dataset types (audio, time series, keypoints)
* More trainers and export formats (COCO, Pascal VOC)
* Tests, translations, UI polish, accessibility

## Guidelines

* Open an issue before big changes.
* Keep it dependency-light; heavy libraries must stay optional.
* Test every dataset type you touch, ideally with the sample datasets.
* Never commit user data (`workspace/projects/`) or model files.
* Sample data must be synthetic or openly licensed - no personal or private images.

By contributing you agree your work is released under AGPL-3.0.
