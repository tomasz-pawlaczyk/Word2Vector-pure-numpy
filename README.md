# Word2Vec in pure NumPy

**Word2Vec in pure NumPy** is a from-scratch implementation of the Skip-Gram with Negative Sampling (SGNS) algorithm that learns dense vector representations of words directly from raw text. The goal is to train word embeddings where semantically similar words are located close to each other in vector space.

The project demonstrates end-to-end understanding of word embedding training — from text preprocessing and skip-gram pair generation to manual loss computation, gradient updates, and semantic inference using cosine similarity.

# Project Overview

## 1. BBC Articles — Semantic Similarity

The model was trained on a corpus of BBC news articles to verify whether it learns meaningful semantic relationships between words in real-world text.

**Example results:**

- economy → industry, country, market, bank  
- president → chairman, director, executive, chief  
- police → authorities, ministers, officials  
- election → general, campaign, latest  

These results show that the model captures topical and functional similarity rather than simple word co-occurrence.

![BBC results](images/bbc.png)

 `data/bbc`

---

## 2. Vector Mathematics — Word Analogies

To validate the geometric properties of the embedding space, the model was trained on a large Wikipedia corpus and tested on classic word analogies.

**Observed relationship:**

$$
\textbf{king} - \textbf{man} + \textbf{woman} \approx \textbf{queen}
$$

This confirms that the learned embeddings encode semantic directions and linear relationships in vector space — a key property of correctly trained Word2Vec models.

![Vector math](images/vector_math.png)

`data/wikitext`

---

## 3. Shakespeare vs Reddit — Dataset Influence on Word Meaning

To show how embeddings depend on the training data, the model was trained on two very different corpora: Shakespeare’s works and Reddit conversations.

**Query word:** `love`

- **Shakespeare corpus:**  
  honour, hate, life, yours, joy  

- **Reddit conversations:**  
  sex, fucking, fuck, loved, girls, kissing  

The contrast demonstrates that Word2Vec learns meaning from context distribution — poetic and emotional in Shakespeare, more casual and physical in Reddit.

![Shakespeare vs Reddit](images/shakespeare_vs_reddit.png)

``data/shakespeare`` & ``data/reddit``

# Mathematical Background

This project implements the full training loop of **Word2Vec Skip-Gram with Negative Sampling (SGNS)** entirely in pure NumPy. All key components — forward pass, loss, gradients, and SGD updates — are derived and coded manually without autograd or deep learning frameworks.

---

## Core Idea

The goal of Word2Vec is to learn vector representations of words such that words appearing in similar contexts are close in vector space.

Each word has two vectors:

- input embedding (center word)  
- output embedding (context word)  

For a real pair of words, we want their vectors to have **high dot-product similarity**. For random pairs, we want the similarity to be low.

---

## Skip-Gram with Negative Sampling

For each observed pair $(target, context)$ the model:

1. increases similarity between the real pair  
2. samples $K$ random words (negative samples)  
3. decreases similarity for those random pairs  

The similarity is measured by the dot product passed through sigmoid:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This avoids the expensive full softmax and makes training efficient.

---

## Training Step

For every skip-gram pair the implementation:

- computes dot products using NumPy  
- evaluates the negative sampling loss  
- computes gradients analytically  
- updates embeddings with SGD  

All operations are done directly on vectors — no automatic differentiation is used.

---

## Parameter Updates

Embeddings are updated using standard stochastic gradient descent:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

where $\eta$ is the learning rate.

Training is fully reproducible via a fixed random seed.





# Project Structure

**Directory tree**

```
├── data/  
├── experiments/  
│  └── saved_experiments/  
├── images/  
├── word2vec/  
│  ├── dataset.py  
│  ├── model.py  
│  ├── train.py  
│  ├── query.py  
│  └── utils.py  
├── requirements.txt  
└── README.md
```

**Module summary**

- **dataset.py** — text preprocessing, vocabulary building, skip-gram pairs

- **model.py** — Skip-Gram with Negative Sampling implementation

- **train.py** — end-to-end training and artifact saving

- **query.py** — semantic queries on trained embeddings

- **utils.py** — unigram distribution, negative sampling, similarity tools

---

# Installation & How to Run

## Requirements

- Python 3.9+

- NumPy

- tqdm

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training

```
python -m word2vec.train
```

---

## Query (Inference)

```
python -m word2vec.query
```
