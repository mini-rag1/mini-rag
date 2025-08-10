from enum import Enum

class ResponseSignal(Enum):

    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    PROCESS_SUCESS = "process_success"
    PROCESS_FAILED = "process_failed"
    NO_FILES_ERROR = "no_files_error"
    INSERT_INTO_VECTOR_DB_SUCCESS = "insert_into_vector_db_success"
    INSERT_INTO_VECTOR_DB_ERROR = "insert_into_vector_db_error"
    VECTORDB_COLLECTION_RETRIEVED = "vectordb_collection_retrieved"
    VECTORDB_SEARCH_ERROR = "vectordb_search_error" 
    VECTORDB_SEARCH_SUCCESS = "vectordb_search_success"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
    RAG_ANSWER_ERROR = "rag_answer_error"