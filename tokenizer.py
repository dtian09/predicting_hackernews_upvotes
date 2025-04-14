import re
import torch
from collections import defaultdict

class SimpleSubwordTokenizer:
    def __init__(self):
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.vocab)

    def _basic_tokenize(self, text):
        # Lowercase, then split by whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts):
        # Create a vocab from a list of texts
        for text in texts:
            tokens = self._basic_tokenize(text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.inv_vocab[self.next_id] = token
                    self.next_id += 1

    def tokenize(self, text):
        return self._basic_tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            token_ids = [self.vocab['[CLS]']] + token_ids + [self.vocab['[SEP]']]
        
        return torch.tensor(token_ids, dtype=torch.long)

# Example usage
if __name__ == "__main__":
    texts = [
        "Machine learning is fun!",
        "Deep learning works well for NLP tasks."
    ]

    tokenizer = SimpleSubwordTokenizer()
    tokenizer.build_vocab(texts)

    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_tensor = tokenizer.encode(text)

        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Encoded Tensor: {input_tensor}")
