@echo off
setlocal
rem Resolve all relative paths from the repository root.
pushd "%~dp0..\.."
if errorlevel 1 exit /b 1

rem Edit these paths for your video and trained manual checkpoint.
set "SOURCE=D:\Dre\PDE_yolo_infra\clip\ks_vms_05.mp4"
set "MANUAL_WEIGHTS=D:\Dre\PDE_yolo_infra\model\admin_KS_honeywell_best_evaluation\KS_honeywell_yolo11m.pt"
set "HAND_WEIGHTS=D:\Dre\PDE_yolo_infra\model\hand_yolov8n.pt"
set "ROI=projects\ks_honeywell\roi.json"
set "CONFIG=projects\ks_honeywell\inference_manual_pickup_config.json"
set "OUTPUT=D:\Dre\PDE_yolo_infra\projects\ks_honeywell\output\ks_vms_05_11.mp4"

call conda activate 5090
if errorlevel 1 (
    echo Error: Could not activate Conda environment 5090.
    echo Run this launcher from Anaconda Prompt with Conda available.
    set "RUN_EXIT_CODE=1"
    goto finish
)

if not exist "%ROI%" (
    python projects\ks_honeywell\select_roi.py --source "%SOURCE%" --output "%ROI%"
    if errorlevel 1 (
        echo Error: ROI selection did not complete. Inference cancelled.
        set "RUN_EXIT_CODE=1"
        goto finish
    )
)

python projects\ks_honeywell\inference_manual_pickup.py ^
    --config "%CONFIG%" ^
    --source "%SOURCE%" ^
    --manual-weights "%MANUAL_WEIGHTS%" ^
    --hand-weights "%HAND_WEIGHTS%" ^
    --roi "%ROI%" ^
    --output "%OUTPUT%"
set "RUN_EXIT_CODE=%ERRORLEVEL%"

:finish
popd
pause
endlocal & exit /b %RUN_EXIT_CODE%
