# Models

This directory contains the data models, database schemas, and enumerations that define the structure and types used throughout the Mini-RAG system. The models layer provides abstraction between the database operations and business logic, ensuring data consistency and type safety.

## Overview

The models directory is organized into several key areas:
- **Core Models**: Database interaction classes
- **Database Schemas**: SQLAlchemy and Pydantic schema definitions
- **Enumerations**: Type definitions and constants
- **Data Transfer Objects**: Classes for data validation and serialization

## Directory Structure

```
models/
├── Core Models (Database Interaction)
│   ├── BaseDataModel.py      # Base class for all database models
│   ├── AssetModel.py         # Asset database operations
│   ├── ChunkModel.py         # Document chunk operations
│   └── ProjectModel.py       # Project management operations
│
├── db_schemas/               # Database schema definitions
│   ├── minirag/              # SQLAlchemy schemas (PostgreSQL)
│   │   ├── schemes/
│   │   │   ├── minirag_base.py    # Base SQLAlchemy declarative class
│   │   │   ├── project.py         # Project table schema
│   │   │   ├── asset.py           # Asset table schema
│   │   │   └── data_chunk.py      # Data chunk table schema
│   │   └── alembic/          # Database migration files
│   ├── DataChunk.py          # Pydantic DataChunk model
│   ├── project.py            # Legacy MongoDB project schema
│   └── asset.py              # MongoDB asset schema
│
└── enums/                    # Enumeration definitions
    ├── AssetTypeEnum.py      # Asset type definitions
    ├── DataBaseEnum.py       # Database collection names
    ├── ProcessingEnums.py    # File processing types
    └── ResponseEnums.py      # API response signals
```

## Core Models

### 🔧 BaseDataModel.py

**Purpose**: Abstract base class providing common functionality for all database interaction models.

**Key Features**:
- Database client management
- Configuration access through `get_settings()`
- Shared initialization for all model classes

**Usage**: All other model classes inherit from this base to ensure consistent database connection handling.

### 🗂️ AssetModel.py

**Purpose**: Handles database operations for file assets in the RAG system.

**Key Features**:
- Asset creation and management
- Project-specific asset retrieval
- Asset validation and lookup

**Methods**:
- `create_asset(asset: Asset)`: Creates new asset records
- `get_all_project_assets(project_id, asset_type)`: Retrieves assets by project and type
- `get_asset_record(project_id, asset_name)`: Finds specific asset by name and project

**Usage**: Used by controllers to manage uploaded files and their metadata in the database.

### 📄 ChunkModel.py

**Purpose**: Manages document chunks created during text processing for vector storage.

**Key Features**:
- Bulk chunk insertion with batch processing
- Chunk retrieval with pagination
- Project-based chunk management
- Chunk counting and statistics

**Methods**:
- `create_chunk(chunk: DataChunk)`: Creates individual chunk records
- `insert_many_chunks(chunks, batch_size)`: Bulk insert with batch processing
- `get_project_chunks(project_id, page_no, page_size)`: Paginated chunk retrieval
- `delete_chunks_by_project_id(project_id)`: Cleanup operations
- `get_total_chunks_count(project_id)`: Statistics and pagination support

**Usage**: Critical for the RAG pipeline, storing processed document segments that will be embedded and searched.

### 📁 ProjectModel.py

**Purpose**: Manages RAG projects and their lifecycle.

**Key Features**:
- Project creation and retrieval
- Auto-creation of projects if they don't exist
- Paginated project listing
- Project statistics

**Methods**:
- `create_project(project: Project)`: Creates new project records
- `get_project_or_create_one(project_id)`: Finds or creates projects
- `get_all_projects(page, page_size)`: Paginated project listing with totals

**Usage**: Foundation for organizing all RAG operations by project, ensuring data isolation between different use cases.

## Database Schemas

### SQLAlchemy Schemas (PostgreSQL)

Located in `db_schemas/minirag/schemes/`, these define the actual database table structure:

#### 🏗️ minirag_base.py
- Provides `SQLAlchemyBase` declarative base class
- Foundation for all SQLAlchemy models

#### 📋 project.py - Project Table
**Columns**:
- `project_id`: Primary key (auto-increment)
- `project_uuid`: Unique UUID identifier
- `created_at`, `updated_at`: Timestamp tracking
- **Relationships**: One-to-many with assets and data_chunks

#### 📎 asset.py - Asset Table
**Columns**:
- `asset_id`: Primary key
- `asset_uuid`: Unique identifier
- `asset_project_id`: Foreign key to projects
- `asset_type`, `asset_name`, `asset_size`: Asset properties
- `asset_config`: JSONB field for flexible configuration
- **Relationships**: Many-to-one with project, one-to-many with data_chunks
- **Indexes**: Optimized for project_id and asset_type queries

#### 🧩 data_chunk.py - DataChunk Table
**Columns**:
- `chunk_id`: Primary key
- `chunk_uuid`: Unique identifier
- `chunk_text`: The actual text content
- `chunk_metadata`: JSONB field for flexible metadata
- `chunk_order`: Ordering within document
- `chunk_project_id`, `chunk_asset_id`: Foreign keys
- **Relationships**: Many-to-one with both project and asset
- **Additional**: Includes `RetrievedDocument` Pydantic model for search results

### Pydantic Models

Used for data validation and API serialization:

#### DataChunk.py
- Validates chunk data before database insertion
- Ensures required fields and proper types
- Used in API endpoints and data processing

## Enumerations

### 📂 AssetTypeEnum.py
```python
class AssetTypeEnum(Enum):
    FILE = "file"
```
**Purpose**: Defines supported asset types (extensible for future types like URLs, etc.)

### 🗄️ DataBaseEnum.py
```python
class DataBaseEnum(Enum):
    COLLECTION_PROJECT_NAME = "projects"
    COLLECTION_CHUNK_NAME = "chunks"
    COLLECTION_ASSET_NAME = "assets"
```
**Purpose**: Centralizes database collection/table names for consistency

### ⚙️ ProcessingEnums.py
```python
class ProcessingEnum(Enum):
    TXT = '.txt'
    PDF = '.pdf'
```
**Purpose**: Defines supported file formats for document processing

### 📤 ResponseEnums.py
Comprehensive enumeration of API response signals:
- File operation responses (upload success/failure, type/size errors)
- Processing responses (success/failure)
- Vector database responses (insert, search, retrieval status)
- RAG operation responses (answer generation status)

**Usage**: Provides consistent response codes across the API, making error handling and success tracking standardized.

## Database Migration

The `db_schemas/minirag/alembic/` directory contains:
- Database version control and migration scripts
- Schema evolution tracking
- Environment-specific configurations

**Key Files**:
- `alembic.ini`: Migration configuration
- `env.py`: Migration environment setup
- `versions/`: Individual migration scripts

## Data Flow

```
1. API receives data → Pydantic models validate
2. Controllers process → Core Models handle database operations
3. SQLAlchemy schemas define → Database structure
4. Enums provide → Type safety and constants
5. Models return → Structured data to controllers
```

## Key Relationships

- **Projects** contain multiple **Assets** (files)
- **Assets** are split into multiple **DataChunks** (for processing)
- **DataChunks** are embedded and stored in vector database
- All entities linked via foreign keys for data integrity

## Error Handling

Models implement:
- Database connection error handling
- Transaction management with rollback capabilities
- Validation through Pydantic models
- Batch processing with error recovery
- Consistent response signals through enums

## Configuration

Models access configuration through:
- `BaseDataModel` provides `self.settings` from `get_settings()`
- Database connection strings and parameters
- File size limits and processing parameters
- Vector database configuration

This models layer ensures data consistency, provides type safety, and abstracts database complexity from the business logic layer.
