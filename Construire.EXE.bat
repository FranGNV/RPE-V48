@echo off
title Construction RPE Viewer EXE
echo ============================================
echo       Construction de RPE_Viewer.exe
echo ============================================
echo.
python --version
if errorlevel 1 (
    echo ERREUR : Python n'est pas detecte.
    pause
    exit /b 1
)

python -m PyInstaller --noconfirm --clean --onefile --windowed --name RPE_Viewer app.py

if errorlevel 1 (
    echo.
    echo ============================================
    echo ECHEC DE LA COMPILATION
    echo ============================================
    pause
    exit /b 1
)

if exist "dist\RPE_Viewer.exe" (
    copy /Y "dist\RPE_Viewer.exe" "RPE_Viewer.exe" >nul
)

echo.
echo ============================================
echo Termine !
echo Le fichier RPE_Viewer.exe est dans ce dossier.
echo ============================================
pause
