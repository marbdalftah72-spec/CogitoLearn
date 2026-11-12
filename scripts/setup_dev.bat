@echo off
echo ========================================================
echo   CogitoLearn Environment Quick Setup (Windows)
echo ========================================================

REM Create Virtual Environment
python -m venv venv
call venv\Scripts\activate

REM Upgrade Pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Setup environment file if not exists
if not exist "configs\.env" (
    copy configs\.env.example configs\.env
    echo Created configs\.env from template. Please insert your GOOGLE_API_KEY.
)

echo.
echo Setup completed successfully! Run 'venv\Scripts\activate' to start working.
pause
