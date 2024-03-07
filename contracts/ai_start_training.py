from pydantic import BaseModel

class StartAiTrainingRequest(BaseModel):
    id: str