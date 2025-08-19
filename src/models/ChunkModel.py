from .BaseDataModel import BaseDataModel
from .db_schemas import DataChunk
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy import select,insert,update,delete
from sqlalchemy import func

class ChunkModel(BaseDataModel):
    def __init__(self, db_client):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        return instance

    async def create_chunk(self,chunk : DataChunk):
        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
        return chunk
    
    async def get_chunk(self,chunk_id:str):
        async with self.db_client() as session:
            result = await session.execute(
                select(DataChunk).where(DataChunk.id == chunk_id)
            )
            chunk = result.scalar_one_or_none()

        return chunk

    async def insert_many_chunks(self,chunks:list,batch_size: int = 100):
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0,len(chunks),batch_size):
                    batch = chunks[i:i+batch_size]
                    session.add_all(batch)
            await session.commit()
        return len(chunks)

    async def delete_chunks_by_project_id(self,project_id:int):
        async with self.db_client() as session:
            result = await session.execute(
                delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            )
            await session.commit()
        return result.rowcount

    async def get_project_chunks(self,project_id:int, page_no: int = 1, page_size: int = 50):
        async with self.db_client() as session:
            result = await session.execute(
                select(DataChunk).where(DataChunk.chunk_project_id == project_id)
                .offset((page_no - 1) * page_size)
                .limit(page_size)
            )
            chunks = result.scalars().all()
        return chunks

    async def get_total_chunks_count(self, project_id: int):
        total_count = 0
        async with self.db_client() as session:
            result = await session.execute(
                select(func.count()).select_from(DataChunk).where(DataChunk.chunk_project_id == project_id)
            )
            total_count = result.scalar()
        return total_count