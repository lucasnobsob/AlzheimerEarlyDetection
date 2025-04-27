from typing import Dict
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    stage: str
    probability: str
    all_probabilities: Dict[str, str]