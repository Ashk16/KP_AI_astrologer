@echo off
echo ==========================================
echo   KP AI ASTROLOGER - ROLLBACK TO STABLE
echo ==========================================
echo.
echo This script will rollback to the stable June 30 version
echo Tag: v1.0-stable-june30
echo.
echo WARNING: This will DISCARD all changes made after the stable commit!
echo.
set /p choice="Are you sure you want to continue? (Y/N): "
if /i "%choice%"=="Y" goto rollback
if /i "%choice%"=="Yes" goto rollback
echo Rollback cancelled.
pause
exit /b

:rollback
echo.
echo Rolling back to stable version...
echo.

echo Step 1: Fetching latest from GitHub...
git fetch origin

echo Step 2: Checking out stable tag...
git checkout v1.0-stable-june30

echo Step 3: Creating new branch from stable...
git checkout -b rollback-to-stable

echo Step 4: Force updating main branch to stable...
git checkout main
git reset --hard v1.0-stable-june30

echo.
echo ==========================================
echo   ROLLBACK COMPLETED SUCCESSFULLY!
echo ==========================================
echo.
echo Current state:
git log --oneline -1
echo.
echo Your system is now back to the stable June 30 version.
echo All debilitation framework changes have been removed.
echo.
echo To return to latest version later, run:
echo   git checkout main
echo   git pull origin main
echo.
pause 