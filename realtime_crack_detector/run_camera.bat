@echo off
cd /d %~dp0
REM Real-time detection launcher
python predict_rt.py --cfg configs\config.yaml
