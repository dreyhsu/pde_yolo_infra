@echo off
call conda activate env_ultralytics
python train_yolo.py
pause
