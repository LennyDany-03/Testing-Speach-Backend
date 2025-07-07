from pydantic import BaseModel

class EvaluationRequest(BaseModel):
    transcript: str

class EvaluationResponse(BaseModel):
    feedback: str  # This will be a Markdown string
