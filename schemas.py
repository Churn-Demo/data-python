from pydantic import BaseModel
from typing import Dict, Any

class PredictByIdRequest(BaseModel):
    customer_id: str

class PredictRequest(BaseModel):
    customer_id: str
    features: Dict[str, Any]
