@echo off
"%PREFIX%\python.exe" -I "%PREFIX%\sigma-desktop\install.py"
if errorlevel 1 exit /b 1
exit /b 0
