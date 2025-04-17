''' 
This script implements a Skip-gram model using PyTorch.
input: djb_cbow_dataset_corpus.pt (output of djb_word2vec_tokenise.py)
       djb_cbow_dataset_word_to_idx.pt
       djb_cbow_dataset_idx_to_word.pt
output: skip_gram_embeddings.pt file (dictionary with key= word id, value=embedding
'''
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import wandb

wandb.init(project="skipgram-sgns", config={
    "window_size": 2,
    "embedding_dim": 100,
    "epochs": 100,
    "learning_rate": 0.01,
    "batch_size": 128,
    "training_percentage": 0.7
})

config = wandb.config
window_size = config.window_size
embedding_dim = config.embedding_dim
epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
training_percentage = config.training_percentage

words = torch.load('djb_cbow_dataset_corpus.pt')
word2idx = torch.load('djb_cbow_dataset_word_to_idx.pt')
idx2word = torch.load('djb_cbow_dataset_idx_to_word.pt')
#select a subset of vocabulary
#vocab_size=10000
vocab_size='all vocab'
random.seed(42)

if vocab_size == 'all vocab':
    selected_words = set(word2idx.keys())#unique words of vocab size
    vocab_size = len(selected_words)
else:#Randomly sample a subset of words
    selected_words = set(random.sample(list(word2idx.keys()), vocab_size))#unique words of vocab size
filtered_corpus = [word for word in words if word in selected_words]#corpus of vocab size that include sequence of words and repeats of words
# New vocabulary based only on selected words
filtered_vocab = sorted(set(filtered_corpus))
word2idx = {word: idx for idx, word in enumerate(filtered_vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
print('vocab size:', len(selected_words))

# Generate Skip-gram training pairs
def generate_skipgram_data(words, corpus, window_size):
    pairs = []
    for i in range(0,len(words)):
        target = words[i]
        target_idx = word2idx[target]
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                context_idx = word2idx[corpus[j]]
                pairs.append((target_idx, context_idx))
    return pairs

print('generating skip-gram data')
data_pairs = generate_skipgram_data(list(selected_words), filtered_corpus, window_size)
print('training data of skip-gram is generated')
print('number of training pairs:', len(data_pairs))

print('training data of skip-gram is generated')
print('number of training pairs before split:', len(data_pairs))

# Shuffle and split into training data and testing data
random.shuffle(data_pairs)
split_index = int(training_percentage * len(data_pairs))
train_pairs = data_pairs[:split_index]
test_pairs = data_pairs[split_index:]

print('Training pairs:', len(train_pairs))
print('Test pairs:', len(test_pairs))

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-6)

print('training the skip-gram model')

def batchify(pairs, batch_size):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]

#import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(train_pairs)
    for batch in batchify(train_pairs, batch_size):
        # Unpack batch into separate lists
        target_indices = [pair[0] for pair in batch]
        context_indices = [pair[1] for pair in batch]

        # Convert to tensors and move to device
        target_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)   # [B]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).to(device) # [B]

        # Forward pass: get target embeddings and predict context logits
        target_emb = model.target_embedding(target_tensor)  # [B, D]
        logits = torch.matmul(target_emb, model.context_embedding.weight.t())  # [B, V]

        # CrossEntropyLoss expects raw logits and true class indices
        loss = criterion(logits, context_tensor)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / (len(train_pairs) // batch_size)
    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
    # --- Evaluation on test set ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in batchify(test_pairs, batch_size):
            target_indices = [pair[0] for pair in batch]
            context_indices = [pair[1] for pair in batch]

            target_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)
            context_tensor = torch.tensor(context_indices, dtype=torch.long).to(device)

            target_emb = model.target_embedding(target_tensor)
            logits = torch.matmul(target_emb, model.context_embedding.weight.t())

            loss = criterion(logits, context_tensor)
            test_loss += loss.item()

    avg_test_loss = test_loss / (len(test_pairs) // batch_size)
    wandb.log({"epoch": epoch + 1, "test_loss": avg_test_loss})
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    model.train()  # Back to training mode


# save embeddings to dictionary 
embeddings = {}
device = torch.device("cpu")
model.to(device)
for word in word2idx:
    idx = torch.tensor([word2idx[word]])
    emb = model.target_embedding(idx).detach().numpy()
    embeddings[int(idx)] = emb

# Save the dictionary to a .pt file
torch.save(embeddings, 'skip_gram_embeddings.pt')

#torch.save(model.state_dict(), "skip_gram_model.pt")
#print("Model saved to "skip_gram_model.pt")
