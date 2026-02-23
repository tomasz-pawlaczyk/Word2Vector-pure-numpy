import numpy as np
from collections import Counter
import re


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def build_vocab(tokens, min_count=1):
    counter = Counter(tokens)
    vocab = [w for w, c in counter.items() if c >= min_count]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word


def text_to_indices(tokens, word_to_idx):
    return np.array([word_to_idx[w] for w in tokens if w in word_to_idx], dtype=np.int32)


def generate_skipgram_pairs(indices, window_size):
    pairs = []
    n = len(indices)

    for i in range(n):
        target = indices[i]
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        for j in range(left, right):
            if j != i:
                context = indices[j]
                pairs.append((target, context))

    return pairs
