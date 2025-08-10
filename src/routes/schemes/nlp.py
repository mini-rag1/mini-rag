from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PushRequest(BaseModel):
    do_reset: bool

class SearchRequest(BaseModel):
    text: str
    limit : int = 10