@echo off
echo Stopping Mini-RAG Docker services...
cd "%~dp0"
docker-compose down
echo.
echo Services stopped.
