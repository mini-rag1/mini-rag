#!/bin/bash
set -e 

echo "Running database migrations..."
cd /app/models/db_schemas/minirag/
# Modify alembic.ini to use the correct database URL
sed -i "s|postgresql://postgres:admin@localhost:5432/minirag|postgresql://postgres:admin@pgvector:5432/minirag|g" alembic.ini
alembic upgrade head
cd /app

echo "Starting FastAPI server..."
exec "$@"