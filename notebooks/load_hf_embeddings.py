import torch
from torch.nn.functional import cosine_similarity
import requests
from io import BytesIO

# Download the embeddings file
url = "https://huggingface.co/datasets/dtian09/skip_gram_embeddings/resolve/main/djb_cbow_dataset_idx_to_word.pt"
response = requests.get(url)
embeddings = torch.load(BytesIO(response.content))

print(f"Type of embeddings: {type(embeddings)}")
print(f"Number of words in vocabulary: {len(embeddings)}")

# Test word
test_word = "computer"
if test_word in embeddings:
    test_embedding = embeddings[test_word]
    print(f"\nTesting similarity for word: {test_word}")
    
    # Calculate similarities with all other words
    similarities = []
    for word, embedding in embeddings.items():
        if word != test_word:  # Skip the test word itself
            sim = cosine_similarity(
                test_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            similarities.append((word, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 most similar words
    print(f"\nTop 10 most similar words to '{test_word}':")
    for word, sim in similarities[:10]:
        print(f"{word}: {sim:.4f}")
else:
    print(f"Word '{test_word}' not found in vocabulary") 