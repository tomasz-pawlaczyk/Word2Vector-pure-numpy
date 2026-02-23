import numpy as np


def create_unigram_dist(indices, vocab_size, power=0.75):
    counts = np.bincount(indices, minlength=vocab_size).astype(np.float64)
    probs = counts ** power
    probs /= probs.sum()
    return probs


def sample_negatives(unigram_dist, k):
    return np.random.choice(len(unigram_dist), size=k, p=unigram_dist)




def get_nearest(word, word_to_idx, idx_to_word, embeddings, top_k=5):
    if word not in word_to_idx:
        return []

    w_idx = word_to_idx[word]
    w_vec = embeddings[w_idx]

    norms = np.linalg.norm(embeddings, axis=1)
    sims = embeddings @ w_vec / (norms * np.linalg.norm(w_vec) + 1e-10)

    best = np.argsort(-sims)[1:top_k+1]
    return [idx_to_word[i] for i in best]



def analogy(a, b, c, word_to_idx, idx_to_word, embeddings, top_k=5):
    for w in (a, b, c):
        if w not in word_to_idx:
            return []

    vec = embeddings[word_to_idx[a]] - embeddings[word_to_idx[b]] + embeddings[word_to_idx[c]]

    norms = np.linalg.norm(embeddings, axis=1)
    sims = embeddings @ vec / (norms * np.linalg.norm(vec) + 1e-10)

    best = np.argsort(-sims)[:top_k+3]

    result = []
    for i in best:
        w = idx_to_word[i]
        if w not in {a, b, c}:
            result.append(w)
        if len(result) == top_k:
            break
    return result
