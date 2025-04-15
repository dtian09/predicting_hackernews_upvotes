import torch
from torch.utils.data import Dataset

class SkipGramDataset(Dataset):
    def __init__(self, file_path, window_size=2):
        with open(file_path, "r") as f:
            self.tokens = list(map(int, f.read().split()))
        self.window_size = window_size

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        center = self.tokens[idx]
        context = []
        for i in range(idx - self.window_size, idx + self.window_size + 1):
            if i != idx and 0 <= i < len(self.tokens):
                context.append(self.tokens[i])
        return [(center, ctx) for ctx in context]

def collate_fn(batch):
    centers, contexts = [], []
    for pairs in batch:
        for center, context in pairs:
            centers.append(center)
            contexts.append(context)
    return torch.tensor(centers), torch.tensor(contexts)
