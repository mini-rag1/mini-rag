@echo off
echo Starting Mini-RAG Docker services...
cd "%~dp0"
docker-compose up -d
echo.
echo Services started:
echo  - MongoDB: localhost:27017
echo  - PGVector: localhost:5432
echo.
echo To stop the services, run: docker-compose down
