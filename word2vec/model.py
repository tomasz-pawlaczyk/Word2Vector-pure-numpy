import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SkipGramNS:
    def __init__(self, vocab_size, embed_dim, lr=0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = lr

        self.input_emb = np.random.randn(vocab_size, embed_dim) * 0.01
        self.output_emb = np.random.randn(vocab_size, embed_dim) * 0.01

    def train_step(self, target, context, negatives):
        v = self.input_emb[target]
        u_pos = self.output_emb[context]
        u_neg = self.output_emb[negatives]

        pos_score = sigmoid(np.dot(u_pos, v))
        neg_score = sigmoid(-np.dot(u_neg, v))

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(neg_score + 1e-10))

        grad_pos = (pos_score - 1) * v
        grad_neg = (1 - neg_score)[:, None] * v

        self.output_emb[context] -= self.lr * grad_pos
        self.output_emb[negatives] -= self.lr * grad_neg

        grad_v = (pos_score - 1) * u_pos + np.sum((1 - neg_score)[:, None] * u_neg, axis=0)
        self.input_emb[target] -= self.lr * grad_v

        return loss
