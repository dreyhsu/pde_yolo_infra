@echo off
setlocal
rem Resolve all relative paths from the repository root.
pushd "%~dp0..\.."
if errorlevel 1 exit /b 1

rem Edit these paths for your video and trained manual checkpoint.
set "SOURCE=video\input.mp4"
set "MANUAL_WEIGHTS=logs\ks_honeywell\yolo26m\weights\best.pt"
set "HAND_WEIGHTS=projects\ks_honeywell\weights\hand_yolov8n.pt"
set "OUTPUT=projects\ks_honeywell\output\pickups.mp4"
set "DEVICE=0"

call conda activate 5090
if errorlevel 1 (
    echo Error: Could not activate Conda environment 5090.
    echo Run this launcher from Anaconda Prompt with Conda available.
    set "RUN_EXIT_CODE=1"
    goto finish
)

python projects\ks_honeywell\inference_manual_pickup.py ^
    --source "%SOURCE%" ^
    --manual-weights "%MANUAL_WEIGHTS%" ^
    --hand-weights "%HAND_WEIGHTS%" ^
    --output "%OUTPUT%" ^
    --device "%DEVICE%" ^
    --imgsz 640 ^
    --manual-conf 0.25 ^
    --hand-conf 0.25 ^
    --contact-duration 0.2 ^
    --release-duration 0.3
set "RUN_EXIT_CODE=%ERRORLEVEL%"

:finish
popd
pause
endlocal & exit /b %RUN_EXIT_CODE%
