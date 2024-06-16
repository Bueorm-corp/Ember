def predict(client, text, image_path=None):
    if client.model is None:
        raise ValueError("Model not loaded. Use load_model to load a model first.")
    return client.model.predict(text, image_path)
