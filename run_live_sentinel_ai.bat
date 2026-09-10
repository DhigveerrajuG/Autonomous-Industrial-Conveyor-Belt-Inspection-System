@echo off
title Conveyor Sentinel AI Dashboard
start "" "http://localhost:8000/"
where python >nul 2>&1 && python yolo_sentinel_server.py || "%LocalAppData%\Programs\Python\Python311\python.exe" yolo_sentinel_server.py
pause
