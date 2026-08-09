from src.agents.base import BaseAgent
from src.agents.schemas import RegulatoryComplianceOutput


class RegulatoryComplianceAgent(BaseAgent):
    def run(self, input_data):
        prompt = f"""
        Validate the resident's care plan against the applicable
        state regulatory requirements.

        State:
        {input_data["state"]}

        Care plan:
        {input_data["care_plan"]}

        Identify:
        - whether the care plan is compliant
        - any compliance issues
        - recommendations to address those issues
        """

        return self.harness.generate(prompt,response_schema=RegulatoryComplianceOutput)