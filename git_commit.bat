@echo off
echo ========================================
echo  AquaGenomeAI - Git Commit Helper
echo ========================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in PATH
    echo.
    echo Please install Git for Windows from:
    echo https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo Staging all changes...
git add -A

echo.
echo Files to be committed:
git status --short

echo.
echo ========================================
set /p CONFIRM="Proceed with commit? (y/n): "

if /i "%CONFIRM%" NEQ "y" (
    echo Commit cancelled.
    pause
    exit /b 0
)

echo.
echo Creating commit...
git commit -F COMMIT_MESSAGE.txt

echo.
echo ========================================
echo Commit complete!
echo.
echo Next steps:
echo   1. Review with: git log -1
echo   2. Push with: git push origin main
echo ========================================
echo.
pause
