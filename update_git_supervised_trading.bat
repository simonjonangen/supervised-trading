@echo off
cd /d "%~dp0"

echo Adding files...
git add .

:: Use PowerShell for current date and time
for /f %%i in ('powershell -command "Get-Date -Format yyyy-MM-dd_HH-mm"') do set datetime=%%i
set commitmsg=Auto-update on %datetime%

echo Committing with message: %commitmsg%
git commit -m "%commitmsg%"

echo Pushing to GitHub...
git push

echo Done!
pause

