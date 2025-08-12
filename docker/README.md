# Mini-RAG Docker Setup

This directory contains the Docker configuration for the Mini-RAG project.

## Prerequisites

- Docker Engine
- Docker Compose

## Services

The Docker setup includes the following services:

1. **MongoDB**
   - Port: 27017
   - Username: admin (configurable in .env)
   - Password: admin (configurable in .env)

2. **PGVector (PostgreSQL with vector extension)**
   - Port: 5432
   - Username: postgres
   - Password: admin (configurable in .env)
   - Database: minirag

## Setup Instructions

1. Copy `.env.example` to `.env` if it doesn't exist:

   ```bash
   cp .env.example .env
   ```

2. Modify the `.env` file with your preferred credentials.

3. Start the services:

   **Windows:**
   ```bash
   start-docker.bat
   ```

   **Linux/macOS:**
   ```bash
   ./start-docker.sh
   ```

4. Stop the services:

   **Windows:**
   ```bash
   stop-docker.bat
   ```

   **Linux/macOS:**
   ```bash
   ./stop-docker.sh
   ```

## Data Persistence

All data is persisted in the following directories:

- `./mongodata/`: MongoDB data files
- `./pgdata/`: PostgreSQL data files

## Backup and Restore

To backup your database:

```bash
# Create a backup directory with timestamp
mkdir -p pgdata_backup_$(date +%Y%m%d_%H%M%S)

# Copy PostgreSQL data
cp -r pgdata/* pgdata_backup_$(date +%Y%m%d_%H%M%S)/
```

## Troubleshooting

If you encounter permission issues with the data directories:

```bash
# For Linux/macOS
chmod -R 777 mongodata pgdata
```

If you need to reset the databases completely, stop the containers and delete the data directories:

```bash
docker-compose down
rm -rf mongodata/* pgdata/*
```
