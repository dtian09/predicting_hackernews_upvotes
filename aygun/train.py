import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SkipGramDataset, collate_fn
import pickle

# Load vocabulary
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

vocab_size = len(word_to_idx)
embedding_dim = 100
window_size = 2
batch_size = 512
epochs = 2
lr = 0.01

# Dataset and DataLoader
dataset = SkipGramDataset("encoded.txt", window_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model definition
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        embeds = self.embedding(center_words)
        out = self.output(embeds)
        return out

model = SkipGramModel(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for centers, contexts in loader:
        optimizer.zero_grad()
        output = model(centers)
        loss = loss_fn(output, contexts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}, Loss: {total_loss:.2f}")

# Save trained embedding layer
torch.save(model.embedding.state_dict(), "embeddings.pt")
print("✅ Embeddings saved.")
