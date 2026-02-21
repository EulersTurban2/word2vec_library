import numpy as np
import utils as ut

from pathlib import Path
from model import SkipGram_Model, CBOW_Model

class SAL:
    def __init__(self, model:SkipGram_Model|CBOW_Model, word2idx:dict, idx2word:dict):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word

    def save(self, path:Path, filenames:list = []):
        model_file = "model.npz"
        wordidx_file = "wordidx.pkl"
        if len(filenames) == 2:
            model_file = filenames[0]
            wordidx_file = filenames[1]
        self.model.write_file(path=path, filename=model_file)
        ut.save_vocab(path=path,filename=wordidx_file,word2idx=self.word2idx,idx2word=self.idx2word)

    def return_components(self):
        return self.model, self.word2idx, self.idx2word

    @classmethod
    def open(self,path:Path, filenames:list = []):
        model_file = "model.npz"
        wordidx_file = "wordidx.pkl"
        if len(filenames) == 2:
            model_file = filenames[0]
            wordidx_file = filenames[1]
        model = SkipGram_Model.read_file(path=path,filename=model_file)
        word2idx, idx2word = ut.load_vocab(path=path,filename=wordidx_file)
        return SAL(model=model,word2idx=word2idx,idx2word=idx2word)


if __name__ == "__main__":
    sal = SAL.open(Path("/mnt2/jetbrains_hallucination/models"))
    model, word2idx, idx2word = sal.return_components()