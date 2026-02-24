import numpy as np
import pickle
from utils import get_nearest

emb = np.load("experiments/input_embeddings.npy")

with open("experiments/vocab.pkl", "rb") as f:
    word_to_idx, idx_to_word = pickle.load(f)

top_k = 10 # number of found words

# You can wrrite your words here:
test_words = ["king", "country", "love"]

for w in test_words:
    print(f"\nNearest words to '{w}':")
    print(get_nearest(w, word_to_idx, idx_to_word, emb, top_k))
