# Embedding Provider Migration Guide

This guide explains how to migrate from OpenAI/Cohere embeddings to Langchain Groq/HuggingFace embeddings in the Mini-RAG project.

## Changes Made

1. Added new providers:
   - `LangchainGroqProvider` for text generation (replacing OpenAI)
   - `HuggingFaceEmbeddingProvider` for embeddings (replacing Cohere)

2. Updated dependencies in `requirements.txt`:
   - Added `langchain_groq`
   - Added `langchain_huggingface` 
   - Added `groq`
   - Added `sentence-transformers`

3. Updated configuration:
   - Added support for Groq API key
   - Added support for HuggingFace API key
   - Changed default embedding dimension from 1024 to 384

## Handling Dimension Mismatch

When switching from one embedding provider to another with a different vector dimension (e.g., from Cohere's 1024-dim to HuggingFace's 384-dim), you'll need to migrate your existing vector database.

### Option 1: Automatic Handling

The system has been updated to automatically handle dimension mismatches by:
1. Detecting the mismatch when searching vectors
2. Recreating the collection with the new dimension
3. Returning empty results (since old vectors were deleted)

This approach is useful for testing but requires re-indexing your documents.

### Option 2: Using the Migration Script

For a more controlled migration, use the included migration script:

```bash
python migrate_project.py <project_id>
```

This script will:
1. Get all chunks for the specified project
2. Create/reset the vector collection with the new embedding dimension
3. Re-embed all chunks using the new embedding provider
4. Insert the new vectors into the database

## Troubleshooting

If you encounter dimension mismatch errors like this:

```
ValueError: shapes (98,1024) and (384,) not aligned: 1024 (dim 1) != 384 (dim 0)
```

It means you have existing vectors in the database with a different dimension than your new embedding model. Use the migration script to fix this.

## API Keys

To use the new providers, you'll need to obtain and configure API keys:

1. For Groq: Get an API key from https://console.groq.com/
2. For HuggingFace: Get an API key from https://huggingface.co/settings/tokens

Then update your `.env` file with these keys:

```
GROQ_API_KEY = "your-groq-api-key"
HF_API_KEY = "your-hf-api-key"
```
