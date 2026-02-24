import numpy as np
np.random.seed(42)

from tqdm import tqdm

from dataset import *
from model import SkipGramNS
from utils import *

text = load_text("data/sex.txt")
tokens = tokenize(text)

embed_dim = 100  # size of vector for every word
neg_k = 5        # number of negative samples per positive ones
epochs = 2
window_size = 3  # how far we examine the word

word_to_idx, idx_to_word = build_vocab(tokens, min_count=5)
indices = text_to_indices(tokens, word_to_idx)
pairs = generate_skipgram_pairs(indices, window_size)

vocab_size = len(word_to_idx)

unigram_dist = create_unigram_dist(indices, vocab_size)
model = SkipGramNS(vocab_size, embed_dim)

for epoch in range(epochs):
    total_loss = 0.0

    for target, context in tqdm(pairs):
        negatives = sample_negatives(unigram_dist, neg_k)
        loss = model.train_step(target, context, negatives)
        total_loss += loss

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")

np.save("experiments/input_embeddings.npy", model.input_emb)
np.save("experiments/output_embeddings.npy", model.output_emb)

import pickle
with open("experiments/vocab.pkl", "wb") as f:
    pickle.dump((word_to_idx, idx_to_word), f)

print("\nEmbeddings saved to /experiments/")
