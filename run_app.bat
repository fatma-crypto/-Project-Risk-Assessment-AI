@echo off
echo Starting Project Risk AI...
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Virtual Environment Activated.
echo Launching Streamlit App...
streamlit run app.py
pause
