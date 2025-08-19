from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal
from tqdm.auto import tqdm

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",  # Ensure this prefix matches the URL being accessed
    tags=["api_v1", "nlp"],
)

logger.info("NLP router initialized with prefix '/api/v1/nlp'.")

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):
    logger.info(f"Received request to push index for project: {project_id}")
    logger.info(f"PushRequest data: {push_request.dict()}")  # Log the PushRequest data

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    #creating the collection if not exist
    collection_name = await nlp_controller.create_collection_name(project_id = project.project_id)

    _ = await request.app.vectordb_client.create_collection(collection_name=collection_name,
                                                            embedding_dimension=request.app.embedding_client.embedding_size,
                                                            do_reset=push_request.do_reset)

    #setup batching
    total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    pbar = tqdm(total = total_chunks_count,
                desc = "Vector Indexing Progress",
                position = 0)

    while has_records:
        page_chunks = await chunk_model.get_project_chunks(project_id=project_id, page_no=page_no)
        if len(page_chunks):
            page_no += 1
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids =  list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)
        
        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids
        )

        pbar.update(len(page_chunks))
        inserted_items_count += len(page_chunks)
        
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(
        project=project
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client = request.app.vectordb_client,
        generation_client = request.app.generation_client,
        embedding_client = request.app.embedding_client,
        template_parser = request.app.template_parser
    )

    collection_name = await nlp_controller.create_collection_name(project_id=project.project_id)
    logger.info(f"Searching collection: {collection_name}")

    # Check if the collection exists using get_vector_db_collection_info
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)
    if not collection_info:
        logger.error(f"Collection {collection_name} does not exist.")
        return JSONResponse(
            content={
                "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value,
                "message": f"Collection {collection_name} does not exist."
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )

    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            content={
                "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value,
                "message": "No results found."
            },
            status_code=status.HTTP_400_BAD_REQUEST
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [result.dict() for result in results]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request : Request, project_id: int,search_request : SearchRequest):
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client = request.app.vectordb_client,
        generation_client = request.app.generation_client,
        embedding_client = request.app.embedding_client,
        template_parser = request.app.template_parser
    )

    answer , full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query = search_request.text,
        limit=search_request.limit
    )

    if not answer:
        return JSONResponse(
            status_code = status.HTTP_400_BAD_REQUEST,
            content = {
                "signal":ResponseSignal.RAG_ANSWER_ERROR.value,
                "message":"No answer found."
            }
        )
    
    return JSONResponse(
        status_code = status.HTTP_200_OK,
        content = {
            "signal":ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )