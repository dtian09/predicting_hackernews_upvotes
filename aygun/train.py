import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SkipGramDataset, collate_fn
import pickle
import time

# Load vocabulary
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

vocab_size = len(word_to_idx)
embedding_dim = 100
window_size = 2
batch_size = 128
epochs = 1
lr = 0.01
max_batches = 300  # ⏱️ Optional: limit batches per epoch for faster testing

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
print("✅ Embeddings creation started.")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (centers, contexts) in enumerate(loader):
        optimizer.zero_grad()
        output = model(centers)
        loss = loss_fn(output, contexts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print loss every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save snapshot every 1000 batches
        if batch_idx % 1000 == 0 and batch_idx > 0:
            snapshot_name = f"embeddings_epoch{epoch+1}_batch{batch_idx}.pt"
            torch.save(model.embedding.state_dict(), snapshot_name)
            print(f"📦 Saved intermediate snapshot: {snapshot_name}")

    print(f"✅ Epoch {epoch+1} completed. Total Loss: {total_loss:.2f}")

