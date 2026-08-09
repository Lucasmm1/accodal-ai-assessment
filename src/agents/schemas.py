from pydantic import BaseModel
from typing import List, Optional

class MedicalHistoryOutput(BaseModel):
    summary: str
    key_conditions: list[str]

class RegulatoryComplianceOutput(BaseModel):
    compliant: bool
    issues: list[str]
    recommendations: list[str]

class FamilyCommunicationOutput(BaseModel):
    welcome_summary: str
    important_notes: list[str]

class IntakeResult(BaseModel):
    medical_history: Optional[MedicalHistoryOutput] = None
    compliance: Optional[RegulatoryComplianceOutput] = None
    family_communication: Optional[FamilyCommunicationOutput] = None
    incomplete_agents: List[str] = []