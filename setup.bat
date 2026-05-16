@echo off
REM One-shot setup: venv + deps + llama.cpp clone.
setlocal

if not exist .venv (
    echo Creating virtual env...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements (this may take a few minutes)...
pip install -r requirements.txt

if not exist llama.cpp (
    echo Cloning llama.cpp (shallow) for GGUF conversion...
    git clone --depth 1 https://github.com/ggerganov/llama.cpp llama.cpp
) else (
    echo llama.cpp already present.
)

REM Install the converter's extra deps from llama.cpp/requirements.txt
if exist llama.cpp\requirements.txt (
    echo Installing llama.cpp converter deps...
    pip install -r llama.cpp\requirements.txt
)

echo.
echo ============================================================
echo Setup complete.
echo Activate the venv:   call .venv\Scripts\activate.bat
echo Run the app:         streamlit run app.py
echo ============================================================
endlocal
