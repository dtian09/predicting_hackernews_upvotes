# %% [markdown]
# 
# ### In this section, we are building a tokenizer. To do that, we want to:
# 
#     - create a list of words by spliting text by whitespace
#     - make all words lower case
#     - filter out rare words, that occurred less than N times in the corpus
#     - setting the vocab length to the total number of unique words
# 

# %%
from collections import Counter
import re


def tokenizer(text):
    # split the string into a list of words by whitespace and remove punctuation
    text_no_punctuation = re.sub(r'[^\w\s]', ' ', text)
    list_of_words = text_no_punctuation.split()

    # lower case the words
    lower_case_words = [word.lower() for word in list_of_words]

    # get the unique words using set list
    unique_words = set(lower_case_words)

    # this is much faster than the dictionary comprehension with count()
    all_word_counts = Counter(lower_case_words)

    # Build the vocabulary
    # filter to only include words that appear at least N times
    N = 5  # minimum frequency threshold
    unique_word_counts = {word: count for word, count in all_word_counts.items() if count >= N}

    print('Number of unique words included in word count before filtering: ', len(all_word_counts))
    print('Number of unique words included in word count after filtering: ', len(unique_word_counts))


    # set vocabulary length to the total number of unique words
    vocab_length = len(unique_word_counts)

    # create a corpus of words that are in the vocabulary
    corpus = [word for word in lower_case_words if word in unique_word_counts]

    # create a word to id mapping with the length of ids being the vocab length
    word_to_id = {word: i for i, word in enumerate(unique_word_counts.keys())}

    # create a id to word mapping with the length of ids being the vocab length
    id_to_word = {i: word for i, word in enumerate(unique_word_counts.keys())}

    return word_to_id, id_to_word, corpus










# %%
