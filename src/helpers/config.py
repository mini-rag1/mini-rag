from pydantic_settings import BaseSettings
from pydantic import Extra
from typing import List

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    HUGGINGFACE_API_KEY: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    
    # ==========================LLM Config========================
    GENERATION_BACKEND: str = "COHERE"
    EMBEDDING_BACKEND: str = "COHERE"

    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    COHERE_API_KEY: str 

    GENERATION_MODEL_ID_LITERAL: List[str] = None
    GENERATION_MODEL_ID: str = None 
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None

    INPUT_DEFAULT_MAX_CHARACTERS: int = None
    GENERATION_DEFAULT_MAX_TOKENS: int = None
    GENERATION_DEFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND_LITERAL: List[str] = None
    VECTOR_DB_BACKEND: str 
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: str = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    DEFAULT_LANG : str = "en"
    PRIMARY_LANG : str = "en"

    class Config:
        env_file = ".env"
        env_prefix = ""  # Add this line to remove any prefix for environment variables

def get_settings() -> Settings:
    return Settings()