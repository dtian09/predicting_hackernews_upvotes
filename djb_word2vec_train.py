from collections import Counter
from itertools import chain
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CBOWDataset(Dataset):
    def __init__(self, data, word_to_idx):
        self.data = data
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]
        context_idxs = torch.tensor([self.word_to_idx[word] for word in context_words], dtype=torch.long)
        target_idx = torch.tensor(self.word_to_idx[target_word], dtype=torch.long)
        return context_idxs, target_idx


corpus = torch.load('djb_cbow_dataset_corpus.pt')
word_to_idx = torch.load('djb_cbow_dataset_word_to_idx.pt')
idx_to_word = torch.load('djb_cbow_dataset_idx_to_word.pt')
training_data = torch.load('djb_cbow_dataset_training_data.pt')

dataset = CBOWDataset(training_data, word_to_idx)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)              # [batch_size, context_size, embedding_dim]
        embeds = embeds.mean(dim=1)                         # average context: [batch_size, embedding_dim]
        out = self.linear(embeds)                           # [batch_size, vocab_size]
        return out
    
import torch.optim as optim

# Setup
embedding_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CBOWModel(vocab_size=len(word_to_idx), embedding_dim=embedding_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for context_idxs, target in dataloader:
        context_idxs = context_idxs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(context_idxs)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")