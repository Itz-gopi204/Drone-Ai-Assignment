@echo off
echo ==========================================
echo Drone Security Analyst Agent - Demo
echo ==========================================
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Running curated demo...
echo.

python -m src.main --curated

echo.
echo ==========================================
echo Demo completed. Press any key to exit.
pause >nul
