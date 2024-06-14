import unittest
import tensorflow as tf
import numpy as np
from Ember import Model

class TestModel(unittest.TestCase):
    def setUp(self):
        # Cargar el modelo antes de cada prueba
        self.model = Model('basic_math_model')

    def test_model_loading(self):
        # Verificar que el modelo se haya cargado correctamente
        self.assertIsInstance(self.model.model, tf.keras.Model)

    def test_prediction(self):
        # Verificar que el modelo hace predicciones correctas
        # Realizar predicción para una operación simple
        operacion = 5 + 5
        resultado = self.model.predict(operacion)
        self.assertEqual(resultado.shape, (1, 4))  # Verificar la forma del resultado

    def test_rounding_result(self):
        # Verificar la función de redondeo de resultados
        resultado = np.array([[0.1, 0.2, 0.3, 0.4]])
        resultado_redondeado = self.model.round_result(resultado)
        self.assertEqual(resultado_redondeado, [0.1, 0.2, 0.3, 0.4])

if __name__ == "__main__":
    unittest.main()
