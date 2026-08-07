@echo off
call conda activate env_ultralytics
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python train_yolo26.py
pause
