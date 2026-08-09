from src.agents.regulatory_compliance import RegulatoryComplianceAgent
from src.agents.family_communication import FamilyCommunicationAgent
from src.agents.medical_history import MedicalHistoryAgent
from src.agents.schemas import RegulatoryComplianceOutput
from src.agents.schemas import FamilyCommunicationOutput
from src.agents.schemas import MedicalHistoryOutput
from src.agents.orchestrator import IntakeOrchestrator
from src.harness import LLMHarness
from src.providers.mock import MockProvider


def test_medical_history_agent():
    provider = MockProvider(
        response={
            "summary": "Resident has a history of diabetes.",
            "key_conditions": ["diabetes"],
        }
    )

    harness = LLMHarness(provider)
    agent = MedicalHistoryAgent(harness)

    result = agent.run({
        "clinical_notes": "Resident has a history of diabetes."
    })

    assert isinstance(result, MedicalHistoryOutput)
    assert result.summary == "Resident has a history of diabetes."
    assert result.key_conditions == ["diabetes"]

def test_regulatory_compliance_agent():
    provider = MockProvider(
        response={
            "compliant": True,
            "issues": [],
            "recommendations": [],
        }
    )

    harness = LLMHarness(provider)
    agent = RegulatoryComplianceAgent(harness)

    result = agent.run({
        "state": "California",
        "care_plan": "Resident receives assistance with medication."
    })

    assert isinstance(result, RegulatoryComplianceOutput)
    assert result.compliant is True
    assert result.issues == []
    assert result.recommendations == []

def test_family_communication_agent():
    provider = MockProvider(
        response={
            "welcome_summary": "Welcome to our care community.",
            "important_notes": ["Medication assistance is provided."]
        }
    )

    harness = LLMHarness(provider)
    agent = FamilyCommunicationAgent(harness)

    result = agent.run({
        "resident_info": "Maria is joining the facility.",
        "care_plan": "Resident receives assistance with medication."
    })

    assert isinstance(result, FamilyCommunicationOutput)
    assert result.welcome_summary == "Welcome to our care community."
    assert result.important_notes == ["Medication assistance is provided."]

def test_intake_orchestrator_runs_all_agents():
    provider = MockProvider(
        responses=[
            {
                "summary": "Resident has a history of diabetes.",
                "key_conditions": ["diabetes"],
            },
            {
                "compliant": True,
                "issues": [],
                "recommendations": [],
            },
            {
                "welcome_summary": "Welcome to our care community.",
                "important_notes": ["Medication assistance is provided."],
            },
        ]
    )

    harness = LLMHarness(provider)

    medical_agent = MedicalHistoryAgent(harness)

    # Por enquanto, usamos o mesmo provider/harness apenas para
    # demonstrar a execução do orchestrator.
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
        "care_plan": "Resident receives assistance with medication.",
        "resident_info": "Maria is joining the facility.",
    })

    assert result.medical_history is not None
    assert result.compliance is not None
    assert result.family_communication is not None
    assert result.incomplete_agents == []

def test_intake_orchestrator_continues_when_agent_fails(caplog):
    provider = MockProvider(
        responses=[
            {
                "summary": "Resident has a history of diabetes.",
                "key_conditions": ["diabetes"],
            },
            None,
            None,
            None,
            {
                "welcome_summary": "Welcome to our care community.",
                "important_notes": ["Medication assistance is provided."],
            },
        ],
        fail_on_attempts=[2, 3, 4],
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

    caplog.set_level("ERROR")

    result = orchestrator.run({
        "clinical_notes": "Resident has a history of diabetes.",
        "state": "California",
        "care_plan": "Resident receives assistance with medication.",
        "resident_info": "Maria is joining the facility.",
    })

    assert result.medical_history is not None
    assert result.compliance is None
    assert result.family_communication is not None
    assert result.incomplete_agents == ["compliance"]
    assert "Regulatory compliance agent failed" in caplog.text