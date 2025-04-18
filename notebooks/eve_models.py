# model.py
import torch.nn as nn
import torch
import torch.nn.functional as F

class CBOW(torch.nn.Module):
   def __init__(self, vocab_size, embedding_dim):
       super(CBOW, self).__init__()
       self.embedding = nn.Embedding(vocab_size, embedding_dim)
       self.linear = nn.Linear(embedding_dim, vocab_size)


   def forward(self, inputs):
       embed = self.embedding(inputs)
       embed = embed.mean(dim=1)
       out = self.linear(embed)
       probs = F.log_softmax(out, dim=1)
       return probs

class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features=200, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, inpt):
        out = self.seq(inpt)
        return out
