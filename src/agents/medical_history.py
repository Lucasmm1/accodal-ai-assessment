from src.agents.base import BaseAgent
from src.agents.schemas import MedicalHistoryOutput


class MedicalHistoryAgent(BaseAgent):
    def run(self, input_data):
        prompt = f"""
        Summarize the resident's clinical notes.

        Clinical notes:
        {input_data["clinical_notes"]}

        Return:
        - a concise summary
        - a list of key medical conditions
        """

        return self.harness.generate(prompt,response_schema=MedicalHistoryOutput)