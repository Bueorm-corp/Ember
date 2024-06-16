import google.generativeai as genai
import PIL.Image

class GoogleAI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def load(self, model_name, **params):
        self.model_name = model_name
        self.params = params

    def predict(self, text, image_path=None):
        model = genai.GenerativeModel(self.model_name)
        if image_path:
            img = PIL.Image.open(image_path)
            response = model.generate_content([text, img])
        else:
            chat = model.start_chat(history=[])
            response = chat.send_message(text)
        return response.text
