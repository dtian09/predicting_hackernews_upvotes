import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import time
import os

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

# Training parameters
EMBEDDING_DIM = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
TRAIN_SPLIT = 0.8  # 80% training, 20% testing

print("\nLoading training data...")
training_data = torch.load("./tensors/eve_training_data.pt")
print("Training data loaded successfully")

print("\nLoading word mappings...")
word_to_id = torch.load("./tensors/eve_word_to_id.pt")
id_to_word = torch.load("./tensors/eve_id_to_word.pt")
print("Word mappings loaded successfully")

# create a custom dataset class
class CBOWDataset(Dataset):
    def __init__(self, training_data):
        self.training_data = training_data
        
    def __len__(self):
        return len(self.training_data)  
    
    def __getitem__(self, idx):
        context_words, target_word = self.training_data[idx]
        
        # Convert context words to tensor of word IDs
        if isinstance(context_words, str):
            context_words = [word_to_id[word] for word in context_words.split()]
        elif isinstance(context_words, list):
            context_words = [word_to_id[word] for word in context_words]
            
        # Convert target word to word ID
        if isinstance(target_word, str):
            target_word = word_to_id[target_word]
            
        # Convert to tensors
        context_tensor = torch.tensor(context_words, dtype=torch.long)
        target_tensor = torch.tensor(target_word, dtype=torch.long)
        
        return context_tensor, target_tensor

# Split dataset into train and test
print("\nSplitting dataset into train and test sets...")
dataset = CBOWDataset(training_data)
train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Train set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create data loaders
print("\nCreating data loaders...")
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

print(f"Train loader batches: {len(train_loader)}")
print(f"Test loader batches: {len(test_loader)}")

# create the cbow model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = torch.mean(embeds, dim=1)
        out = self.linear(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def get_embeddings(self):
        return self.embeddings.weight.data

print("\nCreating model...")
vocab_size = len(word_to_id)
model = CBOWModel(vocab_size, EMBEDDING_DIM)
model = model.to(device)
print(f"Model created with vocab_size={vocab_size}, embedding_dim={EMBEDDING_DIM}")

# Training setup
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for context_words, target_word in data_loader:
            context_words = context_words.to(device)
            target_word = target_word.to(device)
            log_probs = model(context_words)
            loss = criterion(log_probs, target_word)
            total_loss += loss.item()
    return total_loss / len(data_loader)

print("\nStarting training...")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")

# Training loop
best_test_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    for batch_idx, (context_words, target_word) in enumerate(progress_bar):
        # Move data to device
        context_words = context_words.to(device)
        target_word = target_word.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        log_probs = model(context_words)
        loss = criterion(log_probs, target_word)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    # Epoch summary
    train_loss = total_loss / len(train_loader)
    test_loss = evaluate(model, test_loader)
    epoch_time = time.time() - start_time
    
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Time: {epoch_time:.2f} seconds')
    
    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        print("Saving best model...")
        model_path = f"./tensors/cbow_model_dim{EMBEDDING_DIM}_batch{BATCH_SIZE}_best.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")

print("\nTraining completed!")
print(f"Final train loss: {train_loss:.4f}")
print(f"Final test loss: {test_loss:.4f}")

# Save final model
print("\nSaving final model...")
model_path = f"./tensors/cbow_model_dim{EMBEDDING_DIM}_batch{BATCH_SIZE}_final.pt"
torch.save(model.state_dict(), model_path)
print(f"Final model saved to {model_path}")

# Save embeddings
print("\nSaving word embeddings...")
embeddings = model.get_embeddings()
embedding_dict = {id_to_word[i]: embeddings[i].cpu().numpy() for i in range(len(id_to_word))}
embedding_path = f"./tensors/word_embeddings_dim{EMBEDDING_DIM}.pt"
torch.save(embedding_dict, embedding_path)
print(f"Word embeddings saved to {embedding_path}")

# Print some example embeddings
print("\nExample embeddings:")
for word in list(embedding_dict.keys())[:5]:
    print(f"{word}: shape {embedding_dict[word].shape}")

