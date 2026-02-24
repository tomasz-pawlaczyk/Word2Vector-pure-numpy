import os
import numpy as np
import pickle
from word2vec.utils import get_nearest

BASE_DIR = os.path.join("experiments", "saved_experiments")


def load_artifacts(dataset_name):
    path = os.path.join(BASE_DIR, dataset_name)
    emb_in = np.load(os.path.join(path, "input_embeddings.npy"))
    with open(os.path.join(path, "vocab.pkl"), "rb") as f:
        word_to_idx, idx_to_word = pickle.load(f)
    return emb_in, word_to_idx, idx_to_word


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def pause():
    input("\nPress ENTER to continue...\n")


# =========================================================
# 1. BBC TEST
# =========================================================
print("\n" + "=" * 60)
print("1. BBC — Semantic Similarity")
print("=" * 60)

emb, w2i, i2w = load_artifacts("bbc")

test_words = ["economy", "president", "police", "election"]

for w in test_words:
    print(f"\nNearest words to '{w}':")
    print(get_nearest(w, w2i, i2w, emb, top_k=5))

pause()



# 2. VECTOR MATHEMATICS (WIKITEXT)
print("\n" + "=" * 60)
print("2. Vector Mathematics — Analogy")
print("=" * 60)

emb, w2i, i2w = load_artifacts("wikitext")

a, b, c, target = "king", "man", "woman", "queen"

va = emb[w2i[a]]
vb = emb[w2i[b]]
vc = emb[w2i[c]]
vq = emb[w2i[target]]

pred_vec = va - vb + vc

print("\nEquation:")
print("king - man + woman ≈ queen")

print("\nFirst 5 dims of predicted vector:")
print(pred_vec[:5])

print("\nFirst 5 dims of queen vector:")
print(vq[:5])

print("\nCosine similarity:")
print(cosine(pred_vec, vq))

pause()


# 3. SHAKESPEARE vs REDDIT
print("\n" + "=" * 60)
print("3. Shakespeare vs Reddit — 'love'")
print("=" * 60)

print("\nShakespeare:")
emb_s, w2i_s, i2w_s = load_artifacts("shakespeare")
print(get_nearest("love", w2i_s, i2w_s, emb_s, top_k=10))

print("\nReddit:")
emb_r, w2i_r, i2w_r = load_artifacts("reddit")
print(get_nearest("love", w2i_r, i2w_r, emb_r, top_k=10))

print("\nDone.")