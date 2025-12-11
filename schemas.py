# schemas.py

from pydantic import BaseModel

class PatientCreate(BaseModel):
    name: str
    birthDate: str
    gender: str

class PatientResponse(BaseModel):
    id: int
    name: str
    birthDate: str
    gender: str
