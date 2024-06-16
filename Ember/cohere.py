import cohere

class CohereAI:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key=api_key)

    def load(self, model_name, **params):
        self.model_name = model_name
        self.params = params

    def predict(self, text, image_path=None):
        stream = self.client.chat_stream(
            model=self.model_name,
            message=text,
            **self.params
        )
        result = ""
        for event in stream:
            if event.event_type == "text-generation":
                result += event.text
        return result
