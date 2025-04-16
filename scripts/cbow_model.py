import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

training_data = torch.load("../data/eve_training_data.pt")

# create a custom dataset class
class CBOWDataset(Dataset):
    def __init__(self, training_data):
        self.training_data = training_data
        
    # overriding the __len__ method to tell PyTorch how many samples you have
    def __len__(self):
        return len(self.training_data)  
    # overriding the __getitem__ method 
    # to tell PyTorch how to retrieve a specific sample and convert it to the format your model expects
    def __getitem__(self, idx):
        context_words, target_word = self.training_data[idx]  # Get a specific sample
        return torch.tensor(context_words), torch.tensor(target_word)  # Convert to tensors

# create a data loader with progress bar
start_time = time.time()
data_loader = DataLoader(CBOWDataset(training_data), 
                         batch_size=10, 
                         shuffle=True)
data_loader = tqdm(data_loader, desc="Loading batches", unit="batch")
end_time = time.time()
print(f"\nDataLoader initialization took {end_time - start_time:.2f} seconds")

# create the cbow model
class CBOWModel(nn.Module):
    # define the architecture of the model
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__() # call super to inherit from nn.Module
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # create an embedding layer
        self.linear = nn.Linear(embedding_dim, vocab_size) # create a linear layer to project embeddings back to vocab size
        
    # define how data flows through the model
    def forward(self, inputs):
        start_time = time.time()
        embeds = self.embeddings(inputs) # convert input words to embeddings
        out = torch.mean(embeds, dim=1) # average the embeddings
        out = self.linear(out) # project embeddings back to vocab size as vector of logits
        log_probs = F.log_softmax(out, dim=1) # apply softmax to get log probabilities
        end_time = time.time()
        print(f"\nForward pass took {end_time - start_time:.2f} seconds")
        return log_probs


# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
