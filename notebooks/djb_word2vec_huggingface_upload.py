import torch
from torch.nn import Embedding
import numpy as np

from wikipedia2vec import Wikipedia2Vec

from pathlib import Path
import json

import tqdm
                                        
model = Wikipedia2Vec.load('/home/danb/Downloads/enwiki_20180420_nolg_100d.pkl.bz2')

words = model.dictionary.words
dim = model.embeddings.shape[1]

words = sorted(words, key=lambda w: w.lower())

# Build vocab and embedding matrix
word2idx = {}
vectors = []

for idx, word in tqdm(enumerate(words)):
    try:
        vec = model.get_word_vector(word)
        word2idx[word] = idx
        vectors.append(vec)
    except KeyError:
        continue

embedding_matrix = torch.tensor(np.vstack(vectors), dtype=torch.float)
embedding_layer = Embedding.from_pretrained(embedding_matrix)

save_path = Path("wiki2vec_model")

# Save weights
torch.save(embedding_layer.state_dict(), save_path / "pytorch_model.bin")

# Save vocab
with open(save_path / "vocab.json", "w") as f:
    json.dump(word2idx, f)