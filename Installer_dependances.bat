@echo off
title Installation RPE Viewer
echo ============================================
echo      Installation des dependances
echo ============================================
echo.
python --version
if errorlevel 1 (
    echo.
    echo ERREUR : Python n'est pas detecte dans le PATH.
    echo Installe Python puis coche "Add Python to PATH".
    pause
    exit /b 1
)
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
echo.
echo ============================================
echo Installation terminee.
echo Tu peux maintenant lancer Construire.EXE.bat
echo ============================================
pause
