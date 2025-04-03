from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Extra

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    HUGGINGFACE_API_KEY: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    class Config:
        env_file = ".env"
        env_prefix = ""  # Add this line to remove any prefix for environment variables

def get_settings() -> Settings:
    return Settings()