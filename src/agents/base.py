from abc import ABC, abstractmethod
from src.harness import LLMHarness

class BaseAgent(ABC):
    def __init__(self, harness: LLMHarness):
        self.harness = harness

    @abstractmethod
    def run(self, input_data):
        pass