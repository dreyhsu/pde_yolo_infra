@echo off
call conda activate 5090
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python train_yolo26.py
pause
