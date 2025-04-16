import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SkipGramDataset, collate_fn
import pickle
import numpy as np
import time

# -------------------------------
# 📦 Load vocabulary and setup
# -------------------------------
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

vocab_size = len(word_to_idx)
embedding_dim = 100
window_size = 2
batch_size = 128
epochs = 1
learning_rate = 0.01
neg_samples = 5

# -------------------------------
# ⚙️ Negative Sampler Setup
# -------------------------------
# For simplicity, use uniform distribution (better: use actual word frequencies)
unigram_dist = np.ones(vocab_size)
unigram_dist = unigram_dist ** 0.75
unigram_dist = unigram_dist / unigram_dist.sum()

def sample_negative(batch_size, num_negatives):
    return np.random.choice(
        vocab_size,
        size=(batch_size, num_negatives),
        p=unigram_dist
    )

# -------------------------------
# 🧠 Skip-Gram with Negative Sampling Model
# -------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words, negative_words):
        center = self.input_embeddings(center_words)                 # [B, D]
        context = self.output_embeddings(context_words)              # [B, D]
        negatives = self.output_embeddings(negative_words)           # [B, N, D]
        return center, context, negatives

# -------------------------------
# 📉 Loss Function (Binary Logistic)
# -------------------------------
def negative_sampling_loss(center, context, negatives):
    pos_score = torch.sum(center * context, dim=1)                         # [B]
    neg_score = torch.bmm(negatives.neg(), center.unsqueeze(2)).squeeze() # [B, N]

    loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
    loss += -torch.log(torch.sigmoid(neg_score) + 1e-10).mean()
    return loss

# -------------------------------
# 🧺 Dataset and DataLoader
# -------------------------------
dataset = SkipGramDataset("encoded.txt", window_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# -------------------------------
# 🚀 Training Loop
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramNegSampling(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("✅ Training started...")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (centers, contexts) in enumerate(loader):
        centers = centers.to(device)
        contexts = contexts.to(device)

        negatives = sample_negative(len(centers), neg_samples)
        negatives = torch.tensor(negatives, dtype=torch.long).to(device)

        optimizer.zero_grad()
        center_emb, context_emb, negative_emb = model(centers, contexts, negatives)
        loss = negative_sampling_loss(center_emb, context_emb, negative_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}", flush=True)

        if batch_idx % 1000 == 0 and batch_idx > 0:
            snapshot_name = f"embeddings_epoch{epoch+1}_batch{batch_idx}.pt"
            torch.save(model.input_embeddings.state_dict(), snapshot_name)
            print(f"📦 Snapshot saved: {snapshot_name}", flush=True)

    print(f"✅ Epoch {epoch+1} completed. Total Loss: {total_loss:.2f}")

# Save final model
torch.save(model.input_embeddings.state_dict(), "embeddings_final.pt")
print(f"✅ Training complete. Embeddings saved to embeddings_final.pt")
print(f"🕒 Total time: {time.time() - start_time:.2f} seconds")
