import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

# Hyperparameters
window_size = 2 
embedding_dim = 100 #10
epochs = 5 #100
learning_rate = 0.01

words = torch.load('djb_cbow_dataset_corpus.pt') #words
word2idx = torch.load('djb_cbow_dataset_word_to_idx.pt')
idx2word = torch.load('djb_cbow_dataset_idx_to_word.pt')
vocab = set(words)
vocab_size = len(vocab)

# Generate Skip-gram training pairs
def generate_skipgram_data(words, window_size):
    pairs = []
    for i, target in enumerate(words):
        target_idx = word2idx[target]
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                context_idx = word2idx[words[j]]
                pairs.append((target_idx, context_idx))
    return pairs

print('generating skip-gram training data')
training_pairs = generate_skipgram_data(words, window_size)
print('training data of skip-gram is generated')

# Define Skip-gram model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target_word):
        emb_target = self.target_embedding(target_word)
        return emb_target

    def predict_context(self, target_word):
        target_vec = self.forward(target_word)  # shape: [batch_size, embedding_dim]
        all_context_vecs = self.context_embedding.weight  # shape: [vocab_size, embedding_dim]
        scores = torch.matmul(target_vec, all_context_vecs.t())  # [batch_size, vocab_size]
        return scores

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramModel(vocab_size, embedding_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for target_idx, context_idx in training_pairs:
        target_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)
        context_tensor = torch.tensor([context_idx], dtype=torch.long).to(device)

        scores = model.predict_context(target_tensor)  # [1, vocab_size]
        loss = criterion(scores, context_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "./models/skip_gram_model.pt")
print("Model saved to ./models/skip_gram_model.pt")

# View embeddings
#for word in word2idx:
#    idx = torch.tensor([word2idx[word]])
#    emb = model.target_embedding(idx).detach().numpy()
#    print(f"{word}: {emb}")
