@echo off
title Conveyor Sentinel Custom Image Training
where python >nul 2>&1 && python train_from_user_images.py || "%LocalAppData%\Programs\Python\Python311\python.exe" train_from_user_images.py
pause
