@echo off
if exist "RPE_Viewer.exe" (
    start "" "RPE_Viewer.exe"
) else (
    echo RPE_Viewer.exe n'existe pas encore.
    echo Lance d'abord Construire.EXE.bat
    pause
)
