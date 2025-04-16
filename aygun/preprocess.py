from collections import Counter
import pickle
import re  # Regular expression module

filename = "../data/wiki_text_data.txt"

# Step 1: Read and lowercase the text
with open(filename, "r") as f:
    text = f.read().lower()  # ✅ lowercase here

# Step 2: Remove punctuation and non-alphabetic characters
# Only keep alphabetic characters and spaces, remove numbers and punctuations
text = re.sub(r"[^a-z\s]", "", text)

# Step 3: Split into words
words = text.split()

# Step 4: Build vocabulary (most common words)
vocab_size = 70000
word_counts = Counter(words).most_common(vocab_size - 1)
word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(word_counts)}
word_to_idx["<UNK>"] = 0

# Step 5: Create reverse mapping
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 6: Encode the text
encoded = [word_to_idx.get(word, 0) for word in words]

# Step 7: Save results
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)

with open("idx_to_word.pkl", "wb") as f:
    pickle.dump(idx_to_word, f)

with open("encoded.txt", "w") as f:
    f.write(" ".join(map(str, encoded)))

print("✅ Preprocessing complete (with lowercase and cleaned text).")
