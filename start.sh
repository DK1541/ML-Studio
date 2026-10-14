#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && . .venv/bin/activate
  pip install --upgrade pip && pip install -r requirements.txt
  echo "Optional extras: pip install torch torchvision ultralytics xgboost lightgbm catboost"
else
  . .venv/bin/activate
fi
python run.py
