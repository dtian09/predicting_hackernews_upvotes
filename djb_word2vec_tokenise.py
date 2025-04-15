import re
from collections import Counter
from itertools import chain
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


filename = "./data/wiki_text_data.txt"

text = open(filename, 'r', encoding='utf-8').read().lower()

def tokenise(text,min_count = 5):

    text = text.lower()
    text = re.sub(r'[^a-z ]+', ' ', text)  # Remove punctuation and non-alphabetic characters
    words = text.split()

    # Build vocabulary
    word_freq = Counter(words)
    vocab = {word for word, freq in word_freq.items() if freq >= min_count}
    word_to_idx = {word: i for i, word in enumerate(sorted(vocab))}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    # Filter corpus to only include words in the vocab
    corpus = [word for word in words if word in vocab]

    return corpus, word_to_idx, idx_to_word

corpus, word_to_idx, idx_to_word = tokenise(text)
print("Tokenisation complete")

def generate_cbow_data(corpus, context_size=5):
    data = []
    half_context = context_size // 2
    total = len(corpus) - 2 * half_context

    for i in tqdm(range(half_context, len(corpus) - half_context), desc="Generating CBOW pairs"):
        context = (
            corpus[i - half_context : i] + corpus[i + 1 : i + half_context + 1]
        )
        target = corpus[i]
        data.append((context, target))

    return data

training_data = generate_cbow_data(corpus)

#print(training_data[:10])


import torch

torch.save(corpus, 'djb_cbow_dataset_corpus.pt')
print("Corpus saved")
torch.save(word_to_idx, 'djb_cbow_dataset_word_to_idx.pt')
print("Word to idx saved")
torch.save(idx_to_word, 'djb_cbow_dataset_idx_to_word.pt')
print("Idx to word saved")
torch.save(training_data, 'djb_cbow_dataset_training_data.pt')
print("Training data saved")


