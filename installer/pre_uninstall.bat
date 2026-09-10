@echo off
"%PREFIX%\python.exe" -I "%PREFIX%\sigma-desktop\install.py" --remove-shortcuts
exit /b %errorlevel%
