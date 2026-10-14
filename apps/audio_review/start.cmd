@echo off
cd /d "%~dp0\..\.."
python apps\audio_review\server.py --open
if errorlevel 1 pause
