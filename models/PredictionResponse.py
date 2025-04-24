from typing import Dict
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    stage: str
    probability: float
    all_probabilities: Dict[str, float]