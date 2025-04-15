import re
from collections import Counter
from itertools import chain


filename = "text8"

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

def generate_cbow_data(corpus, window_size = 5):
    data = []
    for i in range(window_size, len(corpus) - window_size):
        context = (
            corpus[i - window_size : i] + corpus[i + 1 : i + window_size + 1]
        )
        target = corpus[i]
        data.append((context, target))
    return data

training_data = generate_cbow_data(corpus)