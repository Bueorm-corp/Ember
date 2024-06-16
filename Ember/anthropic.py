import anthropic

class AnthropicAI:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def load(self, model_name, **params):
        self.model_name = model_name
        self.params = params

    def predict(self, text, image_path=None):
        message = self.client.messages.create(
            model=self.model_name,
            messages=[{"text": text}],
            **self.params
        )
        return message.content
