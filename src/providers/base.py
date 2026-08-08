from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, timeout=None) -> str:
        """
        Send a prompt to an LLM and return its response
        """

        pass