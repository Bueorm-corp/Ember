# Ember/models.py

import tensorflow as tf
import os

class Model:
    def __init__(self, model_name):
        self.model_path = os.path.join(os.path.dirname(__file__), f'{model_name}.h5')
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, input_data):
        # Reshape the input to match the expected shape for the model
        input_data_reshaped = tf.constant([[input_data]], shape=(1, 1))
        result = self.model.predict(input_data_reshaped)
        return result

    def round_result(self, result):
        # Round each element in the array to 1 decimal place
        rounded_results = [round(element, 1) for element in result.flatten()]
        return rounded_results
