from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
import os
import sys

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import get_settings
except ImportError:
    # Fallback for when running the file directly
    def get_settings():
        class MockSettings:
            GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
            COHERE_API_KEY = os.getenv('COHERE_API_KEY', '')
        return MockSettings()
    
def rag_process(query: str) -> str:
    """
    Process queries using RAG (Retrieval-Augmented Generation) to get relevant information from documents.
    """
    try:
        settings = get_settings()
        doc_path = os.path.join("data", "rag.txt")
        persist_dir = os.path.join("data", "chromadb")

        # Check if text file exists
        if not os.path.exists(doc_path):
            return f"Document not found at {doc_path}. Please ensure the file is available in the data directory."

        # LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Embeddings
        try:
            embeddings = CohereEmbeddings(
                model="embed-english-v3.0", 
                cohere_api_key=settings.COHERE_API_KEY
            )
        except Exception as e:
            return f"Failed to initialize embeddings: {str(e)}. Please check your Cohere API key."

        # Load docs
        try:
            if doc_path.endswith(".pdf"):
                loader = PyPDFLoader(doc_path)
            else:
                loader = TextLoader(doc_path)
            docs = loader.load()
        except Exception as e:
            return f"Failed to load document: {str(e)}"

        # Split docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)

        # Always rebuild the vector store to ensure fresh data
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)

        try:
            # Different versions of the Chroma wrapper expect different kwarg names
            # Try the most common ones and fall back if needed.
            try:
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    persist_directory=persist_dir,
                    collection_name="example_collection",
                    embedding=embeddings,
                )
            except TypeError as e1:
                # If embedding kwarg isn't accepted or causes a conflict, try embedding_function
                try:
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        persist_directory=persist_dir,
                        collection_name="example_collection",
                        embedding_function=embeddings,
                    )
                except Exception:
                    # Re-raise the original TypeError for clearer debugging
                    raise e1
        except Exception as e:
            return f"Failed to create vector store: {str(e)}"

        # Retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5},
            search_type="similarity"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Get relevant documents
        relevant_docs = retriever.invoke(query)
        context = format_docs(relevant_docs)

        # Simple QA chain
        prompt_template = """
        You are a helpful assistant. Use the context below to answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        prompt = prompt_template.format(context=context, question=query)
        response = llm.invoke(prompt)

        return response.content.strip()

    except Exception as e:
        return f"RAG process failed: {str(e)}. Please try a different query or check your configuration."
