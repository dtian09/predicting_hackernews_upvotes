import torch

embeddings = torch.load("./data/word_embeddings_dim100.pt",weights_only=False)
id_to_word = torch.load("./data/eve_id_to_word.pt")

print(embeddings)
print(id_to_word.shape)


