@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo.
  echo Core install done. Optional extras, run inside this folder:
  echo    .venv\Scripts\pip install torch torchvision   ^(deep learning image models^)
  echo    .venv\Scripts\pip install ultralytics         ^(YOLO detection / segmentation^)
  echo    .venv\Scripts\pip install xgboost lightgbm catboost
) else (
  call .venv\Scripts\activate.bat
)
python run.py
pause
