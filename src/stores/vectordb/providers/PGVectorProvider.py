from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import VectorDBEnums, DistanceMethodEnums,PgVectorDistanceMethodEnums,PgVectorTableSchemaEnums,PgVectorIndexTypeEnums
from models.db_schemas import RetrievedDocument
import logging
from typing import List
from sqlalchemy import text
import json
from sqlalchemy.sql import text as sql_text

class PGVectorProvider(VectorDBInterface):
    def __init__(self,db_client,default_vector_size:int = 786,
                 distance_method:str = None,index_threshold:int = 0):
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        
        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value

        self.index_threshold = index_threshold

        self.pgvector_table_prefix = PgVectorTableSchemaEnums._PREFIX.value
        self.logger = logging.getLogger("uvicorn")

        self.distance_method = distance_method

        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"

    async def connect(self):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                ))
                await session.commit()

    async def disconnect(self):
        pass

    #collection is the same as table for pg
    async def is_collection_existed(self,collection_name:str) -> bool:
        record = None
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text(f'SELECT * FROM pg_tables WHERE tablename = :collection_name')
                results = await session.execute(list_tbl,{"collection_name" : collection_name})
                record = results.scalar_one_or_none()
        return record
    
    async def list_all_collections(self) -> List:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                results = await session.execute(list_tbl, {"prefix": f"{self.pgvector_table_prefix}"})
                records = results.scalars().all()

        return records
    
    async def get_collection_info(self,collection_name: str) -> dict:
        record = None
        async with self.db_client() as session:
            async with session.begin():
                table_info_sql = sql_text(f"""SELECT schemaname,tablename,tableowner,tablespace,hasindexes
                FROM pg_tables
                WHERE tablename = :collection_name""")

                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")

                table_info = await session.execute(table_info_sql, {"collection_name": collection_name})
                record_count = await session.execute(count_sql)

                table_data = table_info.fetchone()

                if not table_data:
                    return None

                return {
                    "table_info": {
                        "schema": table_data[0],
                        "name": table_data[1],
                        "owner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4]
                    },
                    "record_count":record_count.scalar_one()
                }
    async def delete_collection(self, collection_name: str):
        async with self.db_client() as session:
            async with session.begin():
                self.logger.info(f"Deleting collection: {collection_name}")

                delete_sql = sql_text(f"DROP TABLE IF EXISTS {collection_name}")
                await session.execute(delete_sql)
                await session.commit()
        
        return True

    async def create_collection(self, collection_name: str, 
                               embedding_dimension: int,
                               do_reset: bool = False):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.info(f"Creating collection: {collection_name}")

            async with self.db_client() as session:
                async with session.begin():
                    # Using format for the table name since SQLAlchemy doesn't support binding table names
                    create_sql = f"""
                    CREATE TABLE {collection_name} (
                        {PgVectorTableSchemaEnums.ID.value} bigserial PRIMARY KEY, 
                        {PgVectorTableSchemaEnums.TEXT.value} text,
                        {PgVectorTableSchemaEnums.VECTOR.value} vector({embedding_dimension}), 
                        {PgVectorTableSchemaEnums.METADATA.value} jsonb DEFAULT '{{}}', 
                        {PgVectorTableSchemaEnums.CHUNK_ID.value} integer,
                        FOREIGN KEY ({PgVectorTableSchemaEnums.CHUNK_ID.value}) REFERENCES data_chunks(chunk_id)
                    )"""
                    await session.execute(sql_text(create_sql))
                    await session.commit()

            return True

        return False
    
    async def is_index_existed(self,collection_name:str) -> bool:
        index_name = self.default_index_name(collection_name)

        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text(
                    f"""
                    SELECT 1 
                    FROM pg_indexes
                    WHERE indexname = :index_name
                    AND tablename = :collection_name
                    """
                )
                results = await session.execute(check_sql, {
                    "index_name": index_name, 
                    "collection_name": collection_name
                })

                return bool(results.scalar_one_or_none())

    async def create_vector_index(self,collection_name:str,
                                  index_type:str = PgVectorIndexTypeEnums.HNSW.value):
        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        if is_index_existed:
            return False

        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if records_count < self.index_threshold:
                    return False

                self.logger.info(f"START: Creating vector index for collection: {collection_name}")

                index_name = self.default_index_name(collection_name)
                create_idx_sql = sql_text(f"""CREATE INDEX {index_name} ON {collection_name}
                                            USING {index_type} ({PgVectorTableSchemaEnums.VECTOR.value} {self.distance_method})""")
                await session.execute(create_idx_sql)
                await session.commit()

                self.logger.info(f"END: Creating vector index for collection: {collection_name}")

    async def reset_index(self,collection_name:str,
                                    index_type:str = PgVectorIndexTypeEnums.HNSW.value):
        index_name = self.default_index_name(collection_name)

        async with self.db_client() as session:
            async with session.begin():
                drop_idx_sql = sql_text(f"DROP INDEX IF EXISTS {index_name}")
                await session.execute(drop_idx_sql)
                await session.commit()

        return await self.create_vector_index(collection_name=collection_name,
                                               index_type=index_type)

    async def insert_one(self,collection_name: str,
                   text:str, vector:list,
                   metadata: dict = None,
                   record_id: str = None):
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)

        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist.")

        if not record_id:
            self.logger.error("Record ID is required for PGVector insert.")
            return False

        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(f"""
                    INSERT INTO {collection_name} (
                        {PgVectorTableSchemaEnums.TEXT.value},
                        {PgVectorTableSchemaEnums.VECTOR.value},
                        {PgVectorTableSchemaEnums.METADATA.value},
                        {PgVectorTableSchemaEnums.CHUNK_ID.value}
                    ) VALUES (
                        :text,
                        :vector,
                        :metadata,
                        :chunk_id
                    )
                """)

                await session.execute(insert_sql, {
                    "text": text,
                    "vector": "[" + ",".join([str(v) for v in vector]) + "]",
                    "metadata": json.dumps(metadata) if metadata else "{}",
                    "chunk_id": record_id
                })
                await session.commit()

        await self.create_vector_index(collection_name=collection_name)

        return True

    async def insert_many(self,collection_name: str,
                text:list , vector:list,
                metadata: list = None,
                record_id: list = None,
                batch_size: int = 50):
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist.")

        if len(vector) != len(record_id):
            self.logger.error("Vectors and Record IDs must have the same length.")
            return False

        if not metadata or len(metadata) == 0:
            metadata = [None] * len(text)

        async with self.db_client() as session:
            async with session.begin():
                for i in range(0,len(text),batch_size):
                    batch_texts = text[i:i + batch_size]
                    batch_vectors = vector[i:i + batch_size]
                    batch_metadata = metadata[i:i + batch_size] if metadata else None
                    batch_record_id = record_id[i:i + batch_size] if record_id else None

                    values = []

                    for _text,_vector,_metadata,_record_id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_id):
                        values.append({
                            "text":_text,
                            "vector": "[" + ",".join([str(v) for v in _vector]) + "]",
                            "metadata": json.dumps(_metadata) if _metadata else "{}",
                            "chunk_id": _record_id
                        })

                    batch_insert_sql = sql_text(f"""INSERT INTO {collection_name} (text, vector, metadata, chunk_id) 
                                                VALUES (:text, :vector, :metadata, :chunk_id)""")
                    
                    await session.execute(batch_insert_sql, values)

        await self.create_vector_index(collection_name=collection_name)


        return True

    async def search_by_vector(self,collection_name:str,
                         vector:list,
                         limit: int) -> List[RetrievedDocument]:
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist.")
            return []

        vector = "[" + ",".join([str(v) for v in vector]) + "]"

        async with self.db_client() as session:
            async with session.begin():
                search_sql = sql_text(f"""SELECT {PgVectorTableSchemaEnums.TEXT.value} as text, 1 -({PgVectorTableSchemaEnums.VECTOR.value} <=> :vector) as similarity
                                      FROM {collection_name}
                                      ORDER BY similarity DESC
                                      LIMIT {limit}
                                      """)
                result = await session.execute(search_sql,{"vector":vector})
                records = result.fetchall()

                return [
                    RetrievedDocument(
                        text = record.text,
                        score = record.similarity
                    )
                    for record in records
                ]