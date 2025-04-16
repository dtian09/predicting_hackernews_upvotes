from collections import Counter
import pickle

filename = "../data/wiki_text_data.txt"

# Read raw text8 file
with open(filename, "r") as f:
    text = f.read()

# Split into words
words = text.split()

# Build vocabulary of most common words
vocab_size = 70000
word_counts = Counter(words).most_common(vocab_size - 1)
word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(word_counts)}
word_to_idx["<UNK>"] = 0

# Reverse mapping
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Encode full text as word IDs
encoded = [word_to_idx.get(word, 0) for word in words]

# Save everything
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)

with open("idx_to_word.pkl", "wb") as f:
    pickle.dump(idx_to_word, f)

with open("encoded.txt", "w") as f:
    f.write(" ".join(map(str, encoded)))

print("✅ Preprocessing complete.")
