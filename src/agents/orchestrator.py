from src.agents.family_communication import FamilyCommunicationAgent
from src.agents.medical_history import MedicalHistoryAgent
from src.agents.regulatory_compliance import RegulatoryComplianceAgent
from src.agents.schemas import IntakeResult

from src.logger import logger


class IntakeOrchestrator:
    def __init__(
        self,
        medical_history_agent: MedicalHistoryAgent,
        compliance_agent: RegulatoryComplianceAgent,
        family_agent: FamilyCommunicationAgent,
    ):
        self.medical_history_agent = medical_history_agent
        self.compliance_agent = compliance_agent
        self.family_agent = family_agent

    def run(self, input_data):
        result = IntakeResult()

        try:
            result.medical_history = self.medical_history_agent.run(input_data)
        except Exception:
            logger.error(
                "Medical history agent failed",
                exc_info=True,
            )
            result.incomplete_agents.append("medical_history")

        try:
            result.compliance = self.compliance_agent.run(input_data)
        except Exception:
            logger.error(
                "Regulatory compliance agent failed",
                exc_info=True,
            )
            result.incomplete_agents.append("compliance")

        try:
            result.family_communication = self.family_agent.run(input_data)
        except Exception:
            logger.error(
                "Family communication agent failed",
                exc_info=True,
            )
            result.incomplete_agents.append("family_communication")

        return result