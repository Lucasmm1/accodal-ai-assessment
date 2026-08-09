from src.agents.schemas import IncidentClassification, IncidentReport, IncidentValidation
from src.harness import LLMHarness
from src.incidents.workflow import IncidentWorkflow
from src.providers.mock import MockProvider
from src.agents.schemas import IncidentClassification
from src.incidents.router import RegulatoryRouter

def test_incident_workflow_classifies_incident():
    provider = MockProvider(
        response={
            "incident_type": "fall",
            "regulatory_path": "state_incident_notification",
        }
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness)

    incident = IncidentReport(
        incident_id="INC-001",
        description="Resident fell while walking to the dining room.",
        resident_id="RES-001",
        staff_id="STAFF-001",
        date="2026-08-08",
        location="Dining room",
    )

    result = workflow.classify(incident)

    assert isinstance(result, IncidentClassification)
    assert result.incident_type == "fall"
    assert result.regulatory_path == "state_incident_notification"

def test_regulatory_router_routes_known_incident():
    router = RegulatoryRouter()

    classification = IncidentClassification(
        incident_type="fall",
        regulatory_path="anything",
    )

    result = router.route(classification)

    assert result == "state_incident_notification"

def test_regulatory_router_falls_back_to_manual_review():
    router = RegulatoryRouter()

    classification = IncidentClassification(
        incident_type="unknown_incident",
        regulatory_path="anything",
    )

    result = router.route(classification)

    assert result == "manual_regulatory_review"

def test_incident_workflow_validates_incident():
    provider = MockProvider(
        response={
            "complete": True,
            "missing_fields": [],
        }
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness)

    incident = IncidentReport(
        incident_id="INC-001",
        description="Resident fell.",
        resident_id="RES-001",
        staff_id="STAFF-001",
        date="2026-08-08",
        location="Dining room",
    )

    result = workflow.validate(incident)

    assert isinstance(result, IncidentValidation)
    assert result.complete is True
    assert result.missing_fields == []

def test_incident_workflow_stops_when_validation_converges():
    provider = MockProvider(
        responses=[
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": True,
                "missing_fields": [],
            },
        ]
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness, max_iterations=3)

    incident = IncidentReport(
        incident_id="INC-001",
        description="Resident fell.",
        resident_id="RES-001",
        staff_id="STAFF-001",
        date="2026-08-08",
    )

    validation, audit_trail, human_review = (
        workflow.validate_until_complete(incident)
    )

    assert validation.complete is True
    assert human_review is False
    assert len(audit_trail) == 2
    assert audit_trail[0]["complete"] is False
    assert audit_trail[1]["complete"] is True

def test_incident_workflow_escalates_after_max_iterations():
    provider = MockProvider(
        responses=[
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
        ]
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness, max_iterations=3)

    incident = IncidentReport(
        incident_id="INC-002",
        description="Resident fell.",
        resident_id="RES-002",
        staff_id="STAFF-002",
        date="2026-08-08",
    )

    validation, audit_trail, human_review = (
        workflow.validate_until_complete(incident)
    )

    assert validation.complete is False
    assert human_review is True
    assert len(audit_trail) == 4

    assert audit_trail[0]["iteration"] == 1
    assert audit_trail[1]["iteration"] == 2
    assert audit_trail[2]["iteration"] == 3

    assert audit_trail[3] == {
        "step": "human_review",
        "status": "required",
    }

def test_incident_workflow_runs_end_to_end():
    provider = MockProvider(
        responses=[
            {
                "incident_type": "fall",
                "regulatory_path": "anything",
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": True,
                "missing_fields": [],
            },
        ]
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness)

    incident = IncidentReport(
        incident_id="INC-003",
        description="Resident fell while walking.",
        resident_id="RES-003",
        staff_id="STAFF-003",
        date="2026-08-08",
    )

    result = workflow.run(incident)

    assert result.status == "completed"
    assert result.human_review_required is False
    assert result.iterations == 2

    assert result.classification is not None
    assert result.classification.incident_type == "fall"
    assert result.classification.regulatory_path == (
        "state_incident_notification"
    )

    assert len(result.audit_trail) == 4

def test_incident_workflow_escalates_when_validation_does_not_converge():
    provider = MockProvider(
        responses=[
            {
                "incident_type": "fall",
                "regulatory_path": "anything",
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
            {
                "complete": False,
                "missing_fields": ["location"],
            },
        ]
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness, max_iterations=3)

    incident = IncidentReport(
        incident_id="INC-004",
        description="Resident fell while walking.",
        resident_id="RES-004",
        staff_id="STAFF-004",
        date="2026-08-08",
    )

    result = workflow.run(incident)

    assert result.status == "human_review_required"
    assert result.human_review_required is True
    assert result.iterations == 3

    assert result.classification is not None
    assert result.classification.incident_type == "fall"
    assert result.classification.regulatory_path == (
        "state_incident_notification"
    )

    assert len(result.audit_trail) == 6
    assert result.audit_trail[-1] == {
        "step": "human_review",
        "status": "required",
    }

def test_incident_workflow_generates_unique_execution_id():
    provider = MockProvider(
        responses=[
            {
                "incident_type": "fall",
                "regulatory_path": "anything",
            },
            {
                "complete": True,
                "missing_fields": [],
            },
            {
                "incident_type": "fall",
                "regulatory_path": "anything",
            },
            {
                "complete": True,
                "missing_fields": [],
            },
        ]
    )

    harness = LLMHarness(provider)
    workflow = IncidentWorkflow(harness)

    incident = IncidentReport(
        incident_id="INC-005",
        description="Resident fell.",
        resident_id="RES-005",
        staff_id="STAFF-005",
        date="2026-08-08",
    )

    first = workflow.run(incident)
    second = workflow.run(incident)

    assert first.execution_id
    assert second.execution_id
    assert first.execution_id != second.execution_id