from pydantic import BaseModel,Field,validator
from typing import List, Optional
from bson.objectid import ObjectId #used for the object id used in the db in mongo

class Project(BaseModel):
    id : Optional[ObjectId] = Field(None,alias="_id")
    project_id : str = Field(...,min_length=1)

    @validator('project_id') #custom validation
    def validate_project_id(cls,value):
        if not value.isalnum():
            raise ValueError('Project ID must be alphanumeric')
        return value
    
    #because we are using the objectid from the pydantic library we need to add this config class
    #to make it allow this type and not give us shity errors
    class Config:
        arbitrary_types_allowed = True 

    @classmethod
    def get_indexes(cls):
        return [
            {
                "key": [("project_id",1)],
                "name": "project_id_index_1",
                "unique": False
            }
        ]
    