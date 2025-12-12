@echo off
REM Activate your venv (PowerShell activation won't work in .bat directly so use activate.bat)
call .venv_local\Scripts\activate.bat
streamlit run app.py
pause
