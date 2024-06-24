**EMBER**

#Ember is a library to load and use BueormLLC models in a more efficient and simple way

**Install library**
```bash
pip install ember
```

**Use Library**
```python
import ember
from ember import Ember, Predict

tokenizer = Ember.load_tokenizer('LDM-base')
model = Ember.load_model('LDM-base')

text = 'texto de ejemplo'
predict = Predict(model, tokenizer, text)
print(predict)
```
