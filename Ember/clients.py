from .openai import OpenAI
from .google import GoogleAI
from .cohere import CohereAI
from .anthropic import AnthropicAI

class EmberClient:
    def __init__(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key
        self.model = None

    def load_model(self, model_name, **params):
        if self.provider == "openai":
            self.model = OpenAI(api_key=self.api_key)
        elif self.provider == "google":
            self.model = GoogleAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            self.model = AnthropicAI(api_key=self.api_key)
        elif self.provider == "cohere":
            self.model = CohereAI(api_key=self.api_key)
        else:
            raise ValueError(f"Provider {self.provider} not supported")
        self.model.load(model_name, **params)

    def client(provider, api_key, **kwargs):
        return EmberClient(provider, api_key)
