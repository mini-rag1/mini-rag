from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import VectorDBEnums, DistanceMethodEnums
from qdrant_client import QdrantClient,models
from models.db_schemas import RetrievedDocument
import logging
from typing import List,Union

class QdrantDBProvider(VectorDBInterface):
    async def __init__(self,db_client,default_vector_size:int = 786,
                 distance_method:str = None):
        self.db_client = db_client
        self.client = None
        self.distance_method = distance_method
        self.default_vector_size = default_vector_size

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger("uvicorn")

    async def connect(self):
        self.client = QdrantClient(path = self.db_client)

    async def disconnect(self):
        self.client = None
        self.logger.info("Disconnected from QdrantDB.")

    async def is_collection_existed(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self) -> List:
        return await self.client.get_collections()

    async def get_collection_info(self,collection_name: str) -> dict:
        return await self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self, collection_name: str):
        #validate if collection exists before deleting
        if await self.is_collection_existed(collection_name= collection_name):
            return await self.client.delete_collection(collection_name=collection_name)

    async def create_collection(self, collection_name: str, embedding_dimension: int, do_reset: bool = False):
        """
        Create a collection in the vector database.

        Args:
            collection_name (str): The name of the collection.
            embedding_dimension (int): The dimension of the embeddings.
            do_reset (bool): Whether to reset the collection if it already exists.
        """
        try:
            # Check if the collection exists
            if self.is_collection_existed(collection_name=collection_name):
                if do_reset:
                    self.delete_collection(collection_name=collection_name)
                    self.logger.info(f"Collection {collection_name} deleted.")
                else:
                    self.logger.info(f"Collection {collection_name} already exists. Skipping creation.")
                    return True

            # Create the collection
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_dimension, distance=self.distance_method),
            )
            self.logger.info(f"Collection {collection_name} created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            return False

    async def insert_one(self, collection_name: str,
                   text: str, vector: list,
                   metadata: dict = None,
                   record_id: str = None):
        # validating if collection exists before inserting
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False
        
        try:
            _ = await self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],  # Include record ID
                        vector=vector,
                        payload={
                            "text": text,
                            "metadata": metadata if metadata else {}
                        }
                    )
                ]
            )
            self.logger.info(f"Inserted record with ID {record_id} into collection {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error inserting record into collection {collection_name}: {e}")
            return False

    async def insert_many(self, collection_name: str, vectors: list, payloads: list, record_ids: list = None, batch_size: int = 50):
        """
        Insert multiple records into the vector database.

        Args:
            collection_name (str): The name of the collection.
            vectors (list): List of vectors to insert.
            payloads (list): List of payloads (metadata) to insert.
            record_ids (list): List of unique record IDs.
            batch_size (int): Number of records to insert in each batch.
        """
        if record_ids is None:
            record_ids = [None] * len(vectors)

        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            batch_record_ids = record_ids[i:i + batch_size]

            records = [
                models.Record(
                    id=batch_record_ids[j],  # Include the record ID
                    vector=batch_vectors[j],
                    payload=batch_payloads[j]
                )
                for j in range(len(batch_vectors))
            ]

            try:
                self.client.upload_records(
                    collection_name=collection_name,
                    records=records
                )
            except Exception as e:
                self.logger.error(f"Error inserting batch into collection {collection_name}: {e}")
                return False

        return True
    
    async def search_by_vector(self,collection_name:str,
                         vector:list,
                         limit:int = 5):
        #validating if collection exists before searching
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False
        
        results =  self.client.search(
            collection_name = collection_name,
            query_vector = vector,
            limit = limit,
        )

        if not results or len(results) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "text": result.payload["text"],
                "score": result.score,
            })
            for result in results
        ]






