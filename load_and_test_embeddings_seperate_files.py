import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import sys

from scipy.spatial.distance import cosine

#embeddings = torch.load("../data/word_embeddings_dim100.pt",weights_only=False)

compare_word = "computer"
if len(sys.argv) > 1:
    compare_word = sys.argv[1]
    
#filename = "word_embeddings_dim200_epoch10.pt"
#embeddings_filename = "skip_gram_embeddings.pt"
embeddings_filename = "skip_gram_negative_sampling_embeddings.pt"
embeddings = torch.load(embeddings_filename,weights_only=False)

word_idx_filename = "word2idx.pt"
#word_idx_filename = "djb_cbow_dataset_word_to_idx.pt"
word_idx = torch.load(word_idx_filename,weights_only=False)

idx_word_filename = "idx2word.pt"
#idx_word_filename = "djb_cbow_dataset_idx_to_word.pt"
idx_word = torch.load(idx_word_filename,weights_only=False)

print(f"compare_word: '{compare_word}' on '{word_idx_filename}'")

compare_idx = word_idx[compare_word]
compare_embedding = torch.tensor(embeddings[compare_idx])

print(f"compare_idx: {compare_idx}")
print(f"compare_embedding: {compare_embedding}")

# calculate the cosine similarity between the compare_embedding and all other embeddings
results = [(idx, cosine_similarity(compare_embedding, torch.tensor(value)).item()) 
           for idx, value in embeddings.items()]

# Sort the results by similarity in descending order
results.sort(key=lambda x: x[1], reverse=True)

# print the top 10 results
top=10
print(f"\nTop {top} most similar words to '{compare_word}':")
for idx, similarity in results[:top]:
    print(f"{idx_word[idx]}: {similarity:.4f}")


results.sort(key=lambda x: x[1], reverse=False)
print(f"\nTop {top} most DISSIMILAR words to '{compare_word}':")
for idx, similarity in results[:top]:
    print(f"{idx_word[idx]}: {similarity:.4f}")

# Print the results
#for word, dp in results:
#    print(f"{compare_word} - {word} {dp}")

'''
def find_analogy(a, b, c, embedding_dict, top_n=1):
    """
    Solve analogy: a is to b as c is to ?
    """
    if any(word not in embedding_dict for word in (a, b, c)):
        return None

    # Compute vector for: b - a + c
    vec = embedding_dict[b] - embedding_dict[a] + embedding_dict[c]

    # Find closest words by cosine similarity
    similarities = {}
    for word, emb in embedding_dict.items():
        if word in {a, b, c}:
            continue
        similarities[word] = 1 - cosine(vec, emb)

    # Return top-N most similar words
    return sorted(similarities.items(), key=lambda x: -x[1])[:top_n]

print("\nAnalogy Test Examples:")
examples = [
    ("king", "man", "woman"),
    ("paris", "france", "italy"),
    ("walk", "walking", "swim"),
    ("small", "smaller", "big"),
    ("japan", "sushi", "germany"),
]

for a, b, c in examples:
    result = find_analogy(a, b, c, embeddings)
    if result:
        print(f"{a} : {b} :: {c} : {result[0][0]} (score: {result[0][1]:.4f})")
    else:
        print(f"{a} : {b} :: {c} : ? (missing words in vocab)")
'''
