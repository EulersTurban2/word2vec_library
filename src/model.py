import numpy as np

from pathlib import Path
from utils import NegativeSampler


class SkipGram_Model:
    def __init__(self, vocab_size: int, embedded_size: int = 300):
        self.vocab_size = vocab_size
        self.embedded_size = embedded_size
        # initialize to be random values! (Xavier/Glorot initialization)
        self.W_in = np.random.randn(vocab_size,embedded_size) * np.sqrt(1/embedded_size)
        self.W_out = np.random.randn(embedded_size,vocab_size) * np.sqrt(1/embedded_size) # we change the dimensions so that we dont need to call transposing

    def forward(self, target_idx, context_idx, negative_indices):
        v_w = self.W_in[target_idx] #(,embedded_size)

        all_indices = [context_idx] + negative_indices
        u_vectors = self.W_out[:,all_indices] # (embedded_size, all_indices)

        scores = np.dot(v_w,u_vectors) # (1,all_indices = 1 + len(negative_indices))
        return v_w, u_vectors, scores, all_indices

    def write_file(self, path: Path, filename: str):
        np.savez(path / Path(filename), W_in = self.W_in, W_out = self.W_out, vocab_size = self.vocab_size, embedded_size = self.embedded_size)

    @classmethod
    def read_file(self, path:Path, filename:str):
        data = np.load(path / Path(filename), allow_pickle=False)
    
        vocab_size = int(data["vocab_size"])
        embedded_size = int(data["embedded_size"])
        
        model = SkipGram_Model(vocab_size, embedded_size)
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        
        return model


class CBOW_Model:
    def __init__(self):
        pass
