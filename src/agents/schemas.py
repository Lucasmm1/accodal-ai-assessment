from pydantic import BaseModel, Field
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
    incomplete_agents: List[str] = Field(default_factory=list)

class IncidentReport(BaseModel):
    incident_id: str
    description: str
    resident_id: str
    staff_id: str
    date: str
    location: Optional[str] = None

class IncidentClassification(BaseModel):
    incident_type: str
    regulatory_path: str

class IncidentValidation(BaseModel):
    complete: bool
    missing_fields: List[str]

class IncidentWorkflowOutput(BaseModel):
    execution_id: str
    status: str
    classification: Optional[IncidentClassification] = None
    iterations: int
    human_review_required: bool
    audit_trail: List[dict]