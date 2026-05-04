@echo off
cd /d "%~dp0"
echo Starting Blueberry Yield Prediction website...
echo.
echo Keep this window open while using the website.
echo Website: http://127.0.0.1:8000/
echo.
python -u app.py
pause
