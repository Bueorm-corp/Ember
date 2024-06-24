import unittest
from ember import Ember, Predict

class TestEmber(unittest.TestCase):
    def test_load_tokenizer(self):
        src_vocab, tgt_vocab = Ember.load_tokenizer('LDM-base')
        self.assertIsInstance(src_vocab, dict)
        self.assertIsInstance(tgt_vocab, dict)

    def test_load_model(self):
        model = Ember.load_model('LDM-base')
        self.assertIsNotNone(model)

    def test_predict(self):
        tokenizer = Ember.load_tokenizer('LDM-base')
        model = Ember.load_model('LDM-base')
        text = 'texto de ejemplo'
        prediction = Predict(model, tokenizer, text)
        self.assertIsInstance(str(prediction), str)

if __name__ == '__main__':
    unittest.main()
