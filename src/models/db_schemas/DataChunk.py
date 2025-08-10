from pydantic import BaseModel
from typing import Optional

class DataChunk(BaseModel):
    chunk_text: str
    chunk_metadata: Optional[dict] = None  # Ensure this attribute exists
    chunk_order: int
    chunk_project_id: str
    chunk_asset_id: str
