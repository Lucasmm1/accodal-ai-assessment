from pydantic import BaseModel

class SimpleResponse(BaseModel):
    answer: str