@echo off
echo ========================================
echo AI Surveillance PWA - Theft Detection
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements_pwa.txt

REM Generate alarm sound if it doesn't exist
if not exist "static\sounds\alarm.wav" (
    echo Generating alarm sound...
    python generate_alarm.py
)

REM Start the application
echo.
echo Starting AI Surveillance PWA...
echo Access at: http://localhost:5000
echo.
echo Default login:
echo   Username: admin
echo   Password: admin123
echo.
python app_pwa.py
