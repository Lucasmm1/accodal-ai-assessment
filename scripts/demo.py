from src.agents.schemas import IncidentReport
from src.incidents.workflow import IncidentWorkflow
from src.harness import LLMHarness
from src.providers.mock import MockProvider

from src.agents.medical_history import MedicalHistoryAgent
from src.agents.regulatory_compliance import RegulatoryComplianceAgent
from src.agents.family_communication import FamilyCommunicationAgent
from src.agents.orchestrator import IntakeOrchestrator


def demo_harness_retry():
    print("\n=== DEMO 1: Harness Retry ===")

    provider = MockProvider(
        rate_limit_times=2,
        response="Request succeeded after retries.",
    )

    harness = LLMHarness(
        provider,
        sleep_function=lambda _: None,
    )

    result = harness.generate("Hello")

    print("Final response:", result)
    print("Provider attempts:", provider.attempts)


def demo_agent_failure():
    print("\n=== DEMO 2: Agent Swarm Failure Isolation ===")

    provider = MockProvider(
        fail_on_attempts=[2, 3, 4],
        responses=[
            {
                "summary": "Resident has a history of diabetes.",
                "key_conditions": ["diabetes"],
            },
            None,
            None,
            None,
            {
                "welcome_summary": (
                    "Maria is joining the facility and will receive "
                    "assistance with medication."
                ),
                "important_notes": [],
            },
        ],
    )

    harness = LLMHarness(provider)

    medical_agent = MedicalHistoryAgent(harness)
    compliance_agent = RegulatoryComplianceAgent(harness)
    family_agent = FamilyCommunicationAgent(harness)

    orchestrator = IntakeOrchestrator(
        medical_history_agent=medical_agent,
        compliance_agent=compliance_agent,
        family_agent=family_agent,
    )

    result = orchestrator.run({
        "clinical_notes": "Resident has a history of diabetes.",
        "state": "California",
        "care_plan": (
            "Resident receives assistance with medication."
        ),
        "resident_info": "Maria is joining the facility.",
    })

    print("Medical history:", result.medical_history)
    print("Compliance:", result.compliance)
    print("Family communication:", result.family_communication)
    print("Incomplete agents:", result.incomplete_agents)
    print("Provider attempts:", provider.attempts)

def demo_incident_loop_guard():
    print("\n=== DEMO 3: Incident Workflow Loop Guard ===")

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

    harness = LLMHarness(
        provider,
        sleep_function=lambda _: None,
    )

    workflow = IncidentWorkflow(
        harness,
        max_iterations=3,
    )

    incident = IncidentReport(
        incident_id="INC-DEMO",
        description="Resident fell while walking.",
        resident_id="RES-DEMO",
        staff_id="STAFF-DEMO",
        date="2026-08-08",
    )

    result = workflow.run(incident)

    print("Status:", result.status)
    print("Iterations:", result.iterations)
    print("Human review required:", result.human_review_required)
    print("Audit trail:")

    for event in result.audit_trail:
        print(" ", event)

    print("Execution ID:", result.execution_id)

if __name__ == "__main__":
    demo_harness_retry()
    demo_agent_failure()
    demo_incident_loop_guard()