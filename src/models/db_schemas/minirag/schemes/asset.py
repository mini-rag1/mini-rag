from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, String, Integer, ForeignKey,func,DateTime
import uuid
from sqlalchemy.dialects.postgresql import UUID,JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import Index

class Asset(SQLAlchemyBase):
    __tablename__ = 'assets'

    asset_id = Column(Integer, primary_key=True,autoincrement=True)
    asset_uuid = Column(UUID(as_uuid=True),
                          default = uuid.uuid4,
                          unique = True,
                          nullable=False)

    asset_project_id = Column(Integer,
                              ForeignKey("projects.project_id"),
                              nullable=False)
    
    asset_type = Column(String,nullable=False)
    asset_name = Column(String,nullable=False)
    asset_size = Column(Integer, nullable=False,default=None)
    asset_config = Column(JSONB,nullable=True) #instead of the dict we used the JSONB

    project = relationship("Project", back_populates="assets") #takes from Project to populate in the assets
    data_chunks = relationship("DataChunk", back_populates="asset")

    __table_args__ = (
        Index('asset_project_id_index', asset_project_id),
        Index('ix_asset_type', asset_type)
    )