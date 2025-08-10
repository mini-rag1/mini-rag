from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, String, Integer, ForeignKey,func,DateTime
import uuid
from sqlalchemy.dialects.postgresql import UUID,JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import Index
from pydantic import BaseModel

class DataChunk(SQLAlchemyBase):
    __tablename__ = "data_chunks"
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_uuid = Column(UUID(as_uuid=True),
                          default=uuid.uuid4,
                          unique=True,
                          nullable=False)

    chunk_text = Column(String, nullable=False)
    chunk_metadata = Column(JSONB, nullable=False)
    chunk_order = Column(Integer, nullable=False)

    chunk_project_id = Column(Integer, 
                              ForeignKey("projects.project_id"), 
                              nullable=False)
    chunk_asset_id = Column(Integer, 
                             ForeignKey("assets.asset_id"), 
                             nullable=False)
    
    created_at = Column(DateTime,
                        server_default=func.now(),
                        nullable=False)
    updated_at = Column(DateTime,
                        server_default=func.now(),
                        onupdate=func.now(),
                        nullable=False)
    
    project = relationship("Project", back_populates="data_chunks")
    asset = relationship("Asset", back_populates="data_chunks")

    __table_args__ = (
        Index('ix_data_chunk_project_id', chunk_project_id),
        Index('ix_data_chunk_asset_id', chunk_asset_id)
    )

class RetrievedDocument(BaseModel):
    text: str
    score : float