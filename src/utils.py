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

def search_by_index(idx2word:dict, id:int):
    return idx2word[id]

def search_by_word(word2idx:dict, id: int):
    return word2idx[id]

# activation functions
def one_sigmoid(x:np.float128,boundary:int = 20):
    return 1/(1+np.exp(-np.clip(x,-boundary,boundary)))

def sigmoid(X:np.array):
    return [one_sigmoid(xi) for xi in X]

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

def cross_entropy_loss(labels:np.array,predictions:np.array, error_margin:float):
    return -np.sum(labels * np.log(error_margin*np.ones((1,len(predictions))) + predictions) + (1 - labels) * np.log(np.ones((1,len(predictions))) - predictions + error_margin*np.ones((1,len(predictions)))))

# negative sampler
class NegativeSampler:
    def __init__(self, tokens, n_samples = 5):
        self.n_samples = n_samples
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t,0) + 1
        
        freqs = np.array(list(counts.values()))**0.75
        self.probs = freqs / np.sum(freqs)
        self.words = list(counts.keys())

    def sample(self, target_idx, context_idx, word2idx):
        neg_indices = []
        while len(neg_indices) < self.n_samples:
            sample_word = np.random.choice(self.words, p=self.probs)
            sample_idx = word2idx[sample_word]

            if sample_idx != target_idx and sample_idx != context_idx:
                neg_indices.append(sample_idx)
        return neg_indices

