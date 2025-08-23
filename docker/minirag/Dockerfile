FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY src/ .

#create directory for alembic ini
RUN mkdir -p models/db_schemas/minirag

COPY docker/minirag/alembic.ini /app/models/db_schemas/minirag/alembic.ini

COPY docker/minirag/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

#command to run the application
CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "8000", "--workers", "4"]