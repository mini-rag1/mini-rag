#!/bin/bash
echo "Stopping Mini-RAG Docker services..."
cd "$(dirname "$0")"
docker-compose down
echo ""
echo "Services stopped."
