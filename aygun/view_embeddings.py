import torch
import pickle

embedding_dim = 100

# Load vocab and reverse mapping
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)

# Load embedding model
model = torch.nn.Embedding(len(word_to_idx), embedding_dim)
model.load_state_dict(torch.load("embeddings.pt"))
model.eval()

# Try a word
word = "king"
idx = word_to_idx.get(word, 0)
vector = model(torch.tensor([idx])).detach().numpy()

print(f"🔍 Embedding for '{word}':")
print(vector)
