from pydantic_settings import BaseSettings
from pydantic import Extra
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    COHERE_API_KEY: str
    TAVILY_API_KEY: str = ""

    class Config:
        # Look for .env file in the project root directory
        env_file = Path(__file__).parent / ".env"
        env_prefix = ""

def get_settings() -> Settings:
    return Settings()