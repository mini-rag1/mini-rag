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

    def create_collection_name(self,project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self,project: Project):
        collection_name = self.create_collection_name(project_id = project.project_id)
        return self.vectordb_client.delete_collection(collection_name = collection_name)
    
    def get_vector_db_collection_info(self,project: Project):
        collection_name = self.create_collection_name(project_id = project.project_id)
        collection_info =  self.vectordb_client.get_collection_info(collection_name = collection_name)

        return json.loads(
            json.dumps(collection_info,default = lambda x : x.__dict__)
        )

    def index_into_vector_db(self, project, chunks, do_reset=False, chunks_ids=None):
        # Step 1: Get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # Step 2: Prepare vectors, payloads, and record IDs
        vectors = []
        payloads = []
        
        # Process embeddings in smaller batches to avoid rate limits
        batch_size = 5  # Process 5 chunks at a time
        import time
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Process each chunk in the batch
            batch_vectors = []
            for chunk in batch_chunks:
                embedding = self.embedding_client.embed_text(
                    text=chunk.chunk_text,
                    document_type=DocumentTypeEnum.DOCUMENT.value
                )
                if embedding:
                    batch_vectors.append(embedding)
                else:
                    self.logger.warning(f"Failed to embed chunk: {chunk.chunk_text[:50]}...")
            
            vectors.extend(batch_vectors)
            
            # Add corresponding payloads
            batch_payloads = [
                {"metadata": chunk.chunk_metadata, "text": chunk.chunk_text}
                for chunk in batch_chunks[:len(batch_vectors)]
            ]
            payloads.extend(batch_payloads)
            
            # Add a small delay between batches to avoid hitting rate limits
            if i + batch_size < len(chunks):
                time.sleep(1)
        
        record_ids = chunks_ids if chunks_ids else [None] * len(vectors)

        # Step 3: Create collection if not exists or reset if required
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_dimension=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        # Step 4: Insert data into the vector database
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            vectors=vectors,
            payloads=payloads,
            record_ids=record_ids
        )

        return True

    def search_vector_db_collection(self,project: Project, text:str,limit:int = 10):
        #step1 : get_collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        #step2 : get text embediign vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        #step3 : search vector db collection
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return False

        return results
    
    def answer_rag_question(self,project: Project, query: str,limit : int = 10):
        answer , full_prompt, chat_history = None, None, None

        #step 1 : retrieve documents from vector db
        retrieved_docs = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit
        ) 

        if not retrieved_docs or len(retrieved_docs) == 0:
            return answer , full_prompt, chat_history
            
        #construct llm prompt
        system_prompt = self.template_parser.get("rag","system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag","document_prompt", {
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