@echo off
echo Starting integrated RAG Course Assistant Backend & React Frontend...
start cmd /k "python main.py"
timeout /t 3 >nul
start http://127.0.0.1:8000
