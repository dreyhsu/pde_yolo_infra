@echo off
setlocal
pushd "%~dp0..\.."
if errorlevel 1 exit /b 1

rem Edit SOURCE, or pass the input video as the first argument.
set "SOURCE=D:\Dre\PDE_yolo_infra\clip\ks_vms_04.mp4"
if not "%~1"=="" set "SOURCE=%~1"
set "ROI=projects\ks_honeywell\roi.json"

call conda activate 5090
if errorlevel 1 (
    echo Error: Could not activate Conda environment 5090. Use Anaconda Prompt.
    set "RUN_EXIT_CODE=1"
    goto finish
)
python projects\ks_honeywell\select_roi.py --source "%SOURCE%" --output "%ROI%"
set "RUN_EXIT_CODE=%ERRORLEVEL%"

:finish
popd
pause
endlocal & exit /b %RUN_EXIT_CODE%
