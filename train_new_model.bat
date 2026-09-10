@echo off
title Conveyor Sentinel Model Training
where python >nul 2>&1 && python train_perfect_conveyor_model.py || "%LocalAppData%\Programs\Python\Python311\python.exe" train_perfect_conveyor_model.py
pause