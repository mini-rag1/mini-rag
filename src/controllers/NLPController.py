from .BaseController import BaseController  # Use relative import
from models.db_schemas.minirag.schemes import Project,DataChunk
from stores.llm.LLMEnums import LLMEnums,DocumentTypeEnum, CohereEnums  # Ensure this is imported
from typing import List
import json
import logging

class NLPController(BaseController):
    def __init__(self , vectordb_client,generation_client,embedding_client,template_parser = None):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        
        # Set up logger
        self.logger = logging.getLogger(__name__)

    async def create_collection_name(self,project_id: str):
        return  f"collection_{project_id}".strip()
    
    async def reset_vector_db_collection(self,project: Project):
        collection_name = await self.create_collection_name(project_id = project.project_id)
        return await self.vectordb_client.delete_collection(collection_name = collection_name)

    async def get_vector_db_collection_info(self,project: Project):
        collection_name = await self.create_collection_name(project_id = project.project_id)
        collection_info =  await self.vectordb_client.get_collection_info(collection_name = collection_name)

        return json.loads(
            json.dumps(collection_info,default = lambda x : x.__dict__)
        )

    async def index_into_vector_db(self, project:Project, chunks: List[DataChunk], 
                                   do_reset: bool =False, chunks_ids: List[int] = None):
        # Step 1: Get collection name
        collection_name = await self.create_collection_name(project_id=project.project_id)

        #step 2: mange items
        texts = [chunk.chunk_text for chunk in chunks]
        metadata = [chunk.chunk_metadata for chunk in chunks]
        vectors = self.embedding_client.embed_text(text=texts, document_type=DocumentTypeEnum.DOCUMENT.value)

        # Step 3: Create collection if not exists or reset if required
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_dimension=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        # Step 4: Insert data into the vector database
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            text=texts,
            metadata=metadata,
            vector=vectors,
            record_id=chunks_ids
        )

        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        # Step 1: Get collection name
        collection_name = await self.create_collection_name(project_id=project.project_id)
        
        # Step 2: Get text embedding vector
        vectors = self.embedding_client.embed_text(text=[text], 
                                                  document_type=DocumentTypeEnum.QUERY.value)
        
        if not vectors or len(vectors) == 0:
            return False
        
        query_vector = vectors[0]
        
        # Step 3: Search vector db collection
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )
        
        if not results or len(results) == 0:
            return False
        
        return results

    async def answer_rag_question(self,project: Project, query: str,limit : int = 10):
        answer , full_prompt, chat_history = None, None, None

        #step 1 : retrieve documents from vector db
        retrieved_docs = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit
        ) 

        if not retrieved_docs or len(retrieved_docs) == 0:
            return answer , full_prompt, chat_history
            
        #construct llm prompt
        system_prompt = self.template_parser.get("rag","system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag","document_prompt", 
            {
                "doc_num":idx + 1,
                "chunk_text": self.generation_client.process_text(doc.text),
            })
            for idx, doc in enumerate(retrieved_docs)
        ])

        footer_prompt = self.template_parser.get("rag","footer_prompt")

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role="system",  # Use the valid role 'system'
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt = full_prompt,
            chat_history = chat_history,
        )

        return answer,full_prompt,chat_history