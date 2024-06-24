import torch
import torch.nn as nn
import pickle
import os
from .utils import download_file_from_github

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        
        src_embed = self.positional_encoding(self.src_embed(src.transpose(0, 1)))
        tgt_embed = self.positional_encoding(self.tgt_embed(tgt.transpose(0, 1)))
        
        output = self.transformer(src_embed, tgt_embed, src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask)
        return self.output_linear(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Ember:
    @staticmethod
    def load_tokenizer(name):
        src_vocab_path = f"{name}_src_vocab.pkl"
        tgt_vocab_path = f"{name}_tgt_vocab.pkl"

        download_file_from_github(src_vocab_path, src_vocab_path)
        download_file_from_github(tgt_vocab_path, tgt_vocab_path)

        with open(src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(tgt_vocab_path, 'rb') as f:
            tgt_vocab = pickle.load(f)

        return src_vocab, tgt_vocab

    @staticmethod
    def load_model(name):
        model_path = f"{name}_model.pth"

        download_file_from_github(model_path, model_path)

        src_vocab, tgt_vocab = Ember.load_tokenizer(name)

        model = TransformerModel(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        return model

class Predict:
    def __init__(self, model, tokenizer, text, device='cpu'):
        self.model = model
        self.src_vocab, self.tgt_vocab = tokenizer
        self.text = text
        self.device = torch.device(device)
        self.max_len = 100  # or any appropriate max length

    def tokenize(self, sentence, vocab):
        return [vocab['<START>']] + [vocab.get(word, vocab['<UNK>']) for word in sentence.split()] + [vocab['<END>']]

    def pad_sequence(self, sequence, max_len):
        return sequence + [0] * (max_len - len(sequence))

    def translate(self):
        tokens = self.tokenize(self.text, self.src_vocab)
        tokens = self.pad_sequence(tokens, self.max_len)
        src = torch.tensor(tokens).unsqueeze(0).to(self.device)  # Add batch dimension
        tgt = torch.tensor([self.tgt_vocab['<START>']]).unsqueeze(0).to(self.device)  # Start with <START> token
        
        with torch.no_grad():
            for _ in range(self.max_len):
                output = self.model(src, tgt)
                next_token = output.argmax(2)[:, -1].item()
                tgt = torch.cat((tgt, torch.tensor([[next_token]]).to(self.device)), dim=1)
                if next_token == self.tgt_vocab['<END>']:
                    break
        
        output_tokens = tgt.squeeze().cpu().numpy()
        output_sentence = ' '.join([list(self.tgt_vocab.keys())[list(self.tgt_vocab.values()).index(tok)] for tok in output_tokens if tok in self.tgt_vocab.values()])
        return output_sentence

    def __str__(self):
        return self.translate()
