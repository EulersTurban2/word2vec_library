# shorthand functions for helping the model work
import numpy as np

# vocabulary building functions
def build_vocab(tokens:list):
    unique_words = sorted(list(set(tokens)))
    word2idx = {word : i for i, word in enumerate(unique_words)}
    idx2word = {i : word for i, word in enumerate(unique_words)}
    vocab_size = len(unique_words)
    return word2idx, idx2word, vocab_size

def to_one_hot(idx, vocab_size):
    v = np.zeros(vocab_size)
    v[idx] = 1
    return v


# activation functions
def sigmoid(x:np.float128,boundary:int):
    return 1/(1+np.exp(-np.clip(x,-boundary,boundary)))

def tanh_act(x:np.float128):
    return np.tanh(x)

# metrics
def euclidean_dist(X:np.array) -> np.float128:
    return np.sqrt(np.dot(X,X))

def cosine_similarity(X:np.array, Y:np.array) -> np.float128:
    euclid_x = euclidean_dist(X)
    euclid_y = euclidean_dist(Y)
    numerator = euclid_x * euclid_y
    return np.dot(X,Y) / numerator

