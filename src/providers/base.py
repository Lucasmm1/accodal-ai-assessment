from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, timeout=None, response_schema=None):
        """
        Send a prompt to an LLM and return its response
        """

        pass