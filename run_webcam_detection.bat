@echo off
title Conveyor Sentinel Webcam Detection
where python >nul 2>&1 && python -m ultralytics predict model="weights\best.pt" source=0 conf=0.03 show=True || "%LocalAppData%\Programs\Python\Python311\python.exe" -m ultralytics predict model="weights\best.pt" source=0 conf=0.03 show=True
pause
