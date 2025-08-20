# Controllers

This directory contains the controller classes that handle the business logic for the Mini-RAG (Retrieval-Augmented Generation) system. Controllers act as intermediaries between the API routes and the data/service layers, implementing the core functionality of the application.

## Overview

The controllers follow a hierarchical structure where `BaseController` provides common functionality that other controllers inherit from. Each controller is responsible for a specific domain of functionality within the RAG system.

## Controllers

### 📋 BaseController.py

**Purpose**: Abstract base class providing common functionality for all controllers.

**Key Features**:
- Configuration management through `get_settings()`
- Directory path management for files and databases
- Utility methods for generating random strings
- Database path creation and validation

**Methods**:
- `generate_random_string(length: int)`: Creates random alphanumeric strings
- `get_database_path(db_name: str)`: Returns database path and creates directory if needed

### 📁 ProjectController.py

**Purpose**: Manages project-related operations and file system organization.

**Key Features**:
- Project directory creation and management
- File path resolution for project-specific storage

**Methods**:
- `get_project_path(project_id)`: Returns project directory path, creates if doesn't exist

**Usage**: Used by other controllers to organize files by project ID, ensuring proper isolation between different RAG projects.

### 📄 DataController.py

**Purpose**: Handles file upload validation and management operations.

**Key Features**:
- File upload validation (type and size)
- Unique filename generation to prevent conflicts
- File name sanitization and cleaning

**Methods**:
- `validate_uploaded_file(file: UploadFile)`: Validates file type and size
- `generate_unique_filepath(orig_file_name, project_id)`: Creates unique file paths
- `get_clean_file_name(orig_file_name)`: Sanitizes filenames by removing special characters

**Usage**: Essential for document ingestion pipeline, ensuring uploaded files meet requirements and are stored safely.

### ⚙️ ProcessController.py

**Purpose**: Processes uploaded documents into chunks suitable for vector storage.

**Key Features**:
- Multi-format document loading (PDF, TXT)
- Text chunking and splitting strategies
- Document preprocessing for RAG pipeline

**Methods**:
- `get_file_loader(file_id)`: Returns appropriate loader based on file extension
- `get_file_content(file_id)`: Loads document content using LangChain loaders
- `process_file_content(file_content, file_id, chunk_size, overlap_size)`: Splits documents into chunks
- `process_simpler_splitter(texts, metadatas, chunk_size)`: Alternative simpler text splitting method

**Supported Formats**:
- `.txt` files using TextLoader
- `.pdf` files using PyMuPDFLoader

### 🧠 NLPController.py

**Purpose**: Core NLP operations for the RAG system including vector database operations and question answering.

**Key Features**:
- Vector database collection management
- Document embedding and indexing
- Semantic search and retrieval
- RAG-based question answering

**Dependencies**:
- Vector database client
- Text generation client (LLM)
- Text embedding client
- Template parser for prompts

**Key Methods**:

#### Vector Database Operations
- `create_collection_name(project_id)`: Generates collection names for projects
- `reset_vector_db_collection(project)`: Resets/deletes vector database collections
- `get_vector_db_collection_info(project)`: Retrieves collection metadata
- `index_into_vector_db(project, chunks, do_reset, chunks_ids)`: Indexes document chunks

#### Search and Retrieval
- `search_vector_db_collection(project, text, limit)`: Performs semantic search
- `answer_rag_question(project, query, limit)`: Complete RAG pipeline for Q&A

**Usage**: This is the heart of the RAG system, orchestrating the entire pipeline from document indexing to answering user questions using retrieved context.

## Architecture Flow

```
1. DataController validates uploaded files
2. ProcessController chunks documents into manageable pieces  
3. NLPController embeds and indexes chunks into vector database
4. NLPController handles user queries by:
   - Embedding the query
   - Retrieving relevant chunks
   - Generating contextual answers using LLM
```

## Dependencies

The controllers rely on several external libraries and internal modules:
- **FastAPI**: For file upload handling
- **LangChain**: For document loading and text splitting
- **Vector Database**: For semantic search capabilities
- **LLM Clients**: For text generation and embeddings
- **Internal Models**: For data schemas and response handling

## Configuration

Controllers inherit configuration through `BaseController` which loads settings from the application's configuration system, including:
- File upload limits and allowed types
- Directory paths for assets and databases
- Database connection settings

## Error Handling

Controllers implement validation and error handling through:
- File validation before processing
- Path existence checks
- Graceful degradation when services are unavailable
- Structured response signals for API communication
