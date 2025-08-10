@echo off
echo Activating Mini-RAG environment with Python 3.11.7...
call %USERPROFILE%\miniconda3\Scripts\activate.bat %USERPROFILE%\miniconda3\envs\mini-rag
echo Environment activated! You can now use python, pip, etc.
echo.
echo Current Python version:
python --version
echo.
echo To run your main app:
echo python src\main.py
cmd /k
