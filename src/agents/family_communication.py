from src.agents.base import BaseAgent
from src.agents.schemas import FamilyCommunicationOutput


class FamilyCommunicationAgent(BaseAgent):
    def run(self, input_data):
        prompt = f"""
        Draft a clear, warm, plain-language welcome summary
        for the resident's family.

        Resident information:
        {input_data["resident_info"]}

        Care information:
        {input_data["care_plan"]}

        The summary should be easy for a family member to understand.
        Avoid unnecessary medical jargon.
        """

        return self.harness.generate(
            prompt,
            response_schema=FamilyCommunicationOutput,
        )