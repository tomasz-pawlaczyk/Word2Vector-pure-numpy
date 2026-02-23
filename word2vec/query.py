import numpy as np
import pickle
from utils import get_nearest, analogy

emb = np.load("experiments/input_embeddings.npy")

with open("experiments/vocab.pkl", "rb") as f:
    word_to_idx, idx_to_word = pickle.load(f)


top_k = 10       # number of found words




# test_words = ["king", "queen", "war", "love", "sword"]
# test_words = ["monster", "science", "friend", "fear"]
# test_words = ["economy", "president", "police", "election", "fear"]

# test_words = ["sex", "alcohol", "blonde", "love", "kissing", "fuck"]
# test_words = ["love", "economy", "plant"]
test_words = ["love"]



for w in test_words:
    print(f"\nNearest words to '{w}':")
    print(get_nearest(w, word_to_idx, idx_to_word, emb, top_k))

# print("\nAnalogy tests:")
#
# tests = [("king", "man", "woman"), ]
#
# for a, b, c in tests:
#     print(f"{a} - {b} + {c} ≈ {analogy(a, b, c, word_to_idx, idx_to_word, emb, 1)[0]}")