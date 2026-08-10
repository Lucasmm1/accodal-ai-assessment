from uuid import uuid4
from src.agents.schemas import (
    IncidentClassification,
    IncidentReport,
    IncidentValidation,
    IncidentWorkflowOutput,
)
from src.incidents.router import RegulatoryRouter
from src.harness import LLMHarness


class IncidentWorkflow:
    def __init__(self, harness: LLMHarness, router=None, max_iterations=3):
        self.harness = harness
        self.router = router or RegulatoryRouter()
        self.max_iterations = max_iterations

    def classify(self, incident: IncidentReport) -> IncidentClassification:
        prompt = f"""
        Classify the following residential care incident.

        Incident description:
        {incident.description}

        Resident:
        {incident.resident_id}

        Determine:
        - the incident type
        - the appropriate regulatory notification path

        Return a structured classification.
        """

        return self.harness.generate(
            prompt,
            response_schema=IncidentClassification,
        )

    def validate(self, incident: IncidentReport) -> IncidentValidation:
        prompt = f"""
        Validate the incident report.

        Incident:
        {incident.model_dump()}

        Check whether all required fields are present.

        Return:
        - whether the report is complete
        - a list of missing fields
        """

        return self.harness.generate(
            prompt,
            response_schema=IncidentValidation,
        )

    def validate_until_complete(self, incident, execution_id=None):
        if execution_id is None:
            execution_id = str(uuid4())
            
        audit_trail = []

        for iteration in range(1, self.max_iterations + 1):
            validation = self.validate(incident)

            audit_trail.append({
                "execution_id": execution_id,
                "step": "validation",
                "iteration": iteration,
                "complete": validation.complete,
                "missing_fields": validation.missing_fields,
            })

            if validation.complete:
                return validation, audit_trail, False

        audit_trail.append({
            "execution_id": execution_id,
            "step": "human_review",
            "status": "required",
        })

        return validation, audit_trail, True

    def run(self, incident: IncidentReport) -> IncidentWorkflowOutput:
        execution_id = str(uuid4())
        audit_trail = []

        classification = self.classify(incident)

        audit_trail.append({
            "execution_id": execution_id,
            "step": "classification",
            "status": "completed",
            "incident_type": classification.incident_type,
        })

        regulatory_path = self.router.route(classification)

        audit_trail.append({
            "execution_id": execution_id,
            "step": "routing",
            "status": "completed",
            "regulatory_path": regulatory_path,
        })

        validation, validation_audit, human_review = (
            self.validate_until_complete(incident, execution_id)
        )

        audit_trail.extend(validation_audit)

        return IncidentWorkflowOutput(
            execution_id=execution_id,
            status="human_review_required" if human_review else "completed",
            classification=IncidentClassification(
                incident_type=classification.incident_type,
                regulatory_path=regulatory_path,
            ),
            iterations=sum(
                1
                for event in validation_audit
                if event["step"] == "validation"
            ),
            human_review_required=human_review,
            audit_trail=audit_trail,
        )