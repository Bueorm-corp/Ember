import openai

class OpenAI:
    def __init__(self, api_key):
        openai.api_key = api_key

    def load(self, model_name, **params):
        self.model_name = model_name

    def predict(self, text, image_path=None):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=text,
            **params
        )
        return response.choices[0].text
