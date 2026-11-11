@echo off
setlocal
call conda activate 5090
rem Allow the existing MLflow file store used by Ultralytics callbacks.
set "MLFLOW_ALLOW_FILE_STORE=true"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python train_yolo26.py
pause
endlocal
