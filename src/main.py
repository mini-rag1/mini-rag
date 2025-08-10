from fastapi import FastAPI
import logging
from routes import base, data, nlp  # Ensure nlp is imported
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from models.ProjectModel import ProjectModel
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from starlette.middleware.base import BaseHTTPMiddleware
from stores.llm.templates.template_parser import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)

app = FastAPI()

async def startup_span():
    logging.info("Starting application setup...")
    settings = get_settings()
    logging.info("Loaded settings.")

    # app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL) # Connect to MongoDB
    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    app.db_engine = create_async_engine(postgres_conn)
    # logging.info("Connected to MongoDB.")
    logging.info("Connected to PostgreSQL database.")
    
    # app.db_client = app.mongo_conn[settings.MONGODB_DATABASE] # Connect to the database
    app.db_client = sessionmaker(
        app.db_engine,
        class_= AsyncSession
        expire_on_commit = False
    )
    logging.info("Connected to the database.")

    app.project_model = ProjectModel(db_client=app.db_client)
    logging.info("Initialized ProjectModel.")

    llm_provider_factory = LLMProviderFactory(config=settings)
    vectordb_provider_factory = VectorDBProviderFactory(config=settings)

    # generation client
    app.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
    logging.info("Initialized generation client.")

    # embedding client
    app.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )
    logging.info("Initialized embedding client.")

    # vectordb client
    app.vectordb_client = vectordb_provider_factory.create(settings.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()
    logging.info("Connected to VectorDB client.")

    #template parser
    app.template_parser = TemplateParser(language=settings.PRIMARY_LANG,
                                         default_language=settings.DEFAULT_LANG)

async def shutdown_span():
    # app.mongo_conn.close()
    app.db_engine.dispose()
    app.vectordb_client.disconnect()

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

app.include_router(base.base_router)
logging.info("Base router loaded.")

app.include_router(data.data_router)
logging.info("Data router loaded.")

app.include_router(nlp.nlp_router)  # Ensure this is included
logging.info("NLP router loaded.")



