from pydantic import BaseModel,Field,validator
from typing import List, Optional
from bson import ObjectId
from datetime import datetime

class Asset(BaseModel):
    #The Asset class defines the properties and validation rules for such resources, 
    #making it easier to work with them programmatically.
    id: Optional[ObjectId] = Field(None,alias='_id')
    asset_project_id: ObjectId
    asset_type: str = Field(...,min_length = 1)
    asset_name: str = Field(...,min_length = 1)
    asset_size : int = Field(ge = 0, default = None)
    asset_config : Optional[dict] = Field(default = None)
    asset_pushed_at: Optional[datetime] = Field(default = datetime.utcnow())

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("asset_project_id",1)],
                "name": "asset_project_id_index_1",
                "unique": False
            },
            {
                "key": [
                    ("asset_project_id",1),
                    ("asset_name",1)
                ],
                "name": "asset_project_id_asset_name_index_1",
                "unique": True
            }
        ]