from dataclasses import dataclass
from enum import Enum

class Framework(Enum):
    ANYSCALE = "anyscale"
    OPENAI = "openai"
    FIREWORKS = "fireworks"
    VERTEXAI = "vertexai"
    SAGEMAKER = "sagemaker"
    PERPLEXITY = "perplexity"
    TOGETHER = "together"
    VLLM = "vllm"

    # helper method to get the list of values/ supported frameworks
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
# One class for all endpoint configs
@dataclass
class EndpointConfig:
    framework: Framework
    api_base: str = None
    api_key: str = None
    model: str = None
    region: str = None #  Used by SageMaker
    endpoint_name: str = None # Used by SageMaker
    project_id: str = None # Used by VertexAI