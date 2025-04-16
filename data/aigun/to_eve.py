import torch

embeddings = torch.load("../data/word_embeddings_dim100.pt",weights_only=False)

print(embeddings.keys())





