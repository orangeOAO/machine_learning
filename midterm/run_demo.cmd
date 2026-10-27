@echo off
echo Open http://localhost:8765 after the models finish loading.
wsl.exe --cd "%~dp0." -e ../.venv/bin/python demo.py
pause
