@echo off
echo ================================================
echo   Video Transcriber Setup - Parakeet TDT 0.6B V3
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if venv exists
if exist "venv" (
    echo [INFO] Virtual environment already exists
) else (
    echo [STEP 1] Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo.
echo [STEP 2] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [STEP 3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [STEP 4] Installing PyTorch with CUDA support...
echo (This may take a few minutes)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [STEP 5] Installing project dependencies...
pip install -r requirements.txt

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo To start transcribing:
echo   1. Activate the environment:  .\venv\Scripts\activate
echo   2. Run the transcriber:       python transcribe.py video.mp4
echo.
echo For more options:  python transcribe.py --help
echo.
pause
