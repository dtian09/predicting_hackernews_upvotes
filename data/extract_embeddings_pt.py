import bz2
import pickle
import torch
from wikipedia2vec import Wikipedia2Vec

# Path to your model file
MODEL_PATH = "enwiki_20180420_nolg_100d.pkl"
OUTPUT_PT = "word_vectors.pt"

# Load the model
#with bz2.open(MODEL_PATH, "rb") as f:
model = Wikipedia2Vec.load(MODEL_PATH)

# Extract word -> vector mapping
word_vectors = {}
for word in model.dictionary.words():
    try:
        word_str = word.text
        vec = model.get_word_vector(word_str)
        word_vectors[word_str] = torch.tensor(vec, dtype=torch.float32)

    except KeyError:
        continue

# Save to .pt file
torch.save(word_vectors, OUTPUT_PT)
print(f"Saved {len(word_vectors)} word vectors to {OUTPUT_PT}")
