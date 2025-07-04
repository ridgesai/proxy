from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="Text to embed")
    run_id: str = Field(..., description="Evaluation run ID")

class GPTMessage(BaseModel):
    role: str
    content: str
    
class InferenceRequest(BaseModel):
    run_id: str = Field(..., description="Evaluation run ID")
    model: Optional[str] = Field(None, description="Model to use for inference")
    temperature: Optional[float] = Field(None, description="Temperature for inference")
    messages: List[GPTMessage] = Field(..., description="Messages to send to the model")