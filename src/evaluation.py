import numpy as np

from model import SkipGram_Model, CBOW_Model

def get_nearest_neighbors(model:SkipGram_Model|CBOW_Model, word:str, word2idx:dict, idx2word:dict, top_k:int = 5) -> list[str]:
    if word not in word2idx:
        raise ValueError("Word not in vocabulary.")
    
    w_idx = word2idx[word]
    
    # target vector
    target_vec = model.W_in[w_idx]  # (d,)
    
    # normalize all embeddings
    embeddings = model.W_in
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)
    
    # normalize target
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-10)
    
    # cosine similarity
    similarities = np.dot(normalized_embeddings, target_vec)
    
    # exclude the word itself
    similarities[w_idx] = -np.inf
    
    # top-k indices
    nearest_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(idx2word[i], similarities[i]) for i in nearest_indices]