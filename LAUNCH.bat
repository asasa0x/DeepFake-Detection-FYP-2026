@echo off
setlocal enabledelayedexpansion
title Deepfake Detection System - Launch
color 0B
chcp 65001 > nul

echo.
echo =========================================================
echo      DEEPFAKE DETECTION SYSTEM  --  AUTO LAUNCHER
echo =========================================================
echo.
echo  [1/3]  Starting Flask AI Backend on port 5000...
echo.

:: Start Flask backend server in a separate window
start "AI Backend Flask :5000" cmd /k "cd /d "%~dp0backend" && py run.py"

:: Wait 4 seconds for Flask to bind to port 5000
timeout /t 4 /nobreak > nul

echo  [2/3]  Starting Frontend HTTP Server on port 3000...
echo.

:: Start Python HTTP server for Frontend in a separate window
start "Frontend Server :3000" cmd /k "cd /d "%~dp0Frontend" && py -m http.server 3000"

:: Wait 2 seconds for frontend server to come online
timeout /t 2 /nobreak > nul

echo  [3/3]  Opening Web App in Chrome...
echo.

:: Try Chrome paths one by one
set "URL=http://127.0.0.1:3000/index.html"
set "CHROME_EXE="

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=C:\Program Files\Google\Chrome\Application\chrome.exe"
    goto :open_chrome
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    goto :open_chrome
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    set "CHROME_EXE=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    goto :open_chrome
)

:: Chrome not found — open default browser
echo  [OK]  Chrome not found. Opening in default browser...
start "" "%URL%"
goto :done

:open_chrome
echo  [OK]  Web App opened in Chrome!
start "" "!CHROME_EXE!" "!URL!"

:done
echo.
echo =========================================================
echo   System Online!
echo.
echo   Frontend  -->  http://127.0.0.1:3000/index.html
echo   Backend   -->  http://127.0.0.1:5000
echo   Health    -->  http://127.0.0.1:5000/health
echo.
echo   To stop: close both server windows.
echo =========================================================
echo.
pause
endlocal
