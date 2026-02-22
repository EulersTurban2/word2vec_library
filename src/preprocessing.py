#Preprocessing functions necessary for further implementation
import re
import numpy as np

from pathlib import Path
from src.utils import build_vocab, search_by_index
from src.utils import NegativeSampler


def build_regex_machine(regex:str) -> re.Pattern:
    return re.compile(regex)

def normalize_line(line:str, regex_machine: re.Pattern, replacment_str):
    norm_line = line.strip()
    norm_line = norm_line.lower()
    norm_line = regex_machine.sub(replacment_str,norm_line)
    return norm_line


def preprocess_list(path:Path, filename:str) -> list[str]:
    
    punc_regex = r'[()*._?!,;:-]+'
    punc_re_machine = build_regex_machine(regex=punc_regex)
    quote_regex = r"`\s*(([^']|\w*'\w)*)'"
    quote_re_machine = build_regex_machine(regex=quote_regex)
    snd_quote_regex = r'"\s*(([^\']|\w*\'\w)*)"'
    snd_quote_re_machine = build_regex_machine(regex=snd_quote_regex)

    full_path = path / filename
    file_lines = []
    with open(full_path,"r") as fd:
        curr_line = ''
        for line in fd:
            cleared_line = line.strip()
            curr_line += cleared_line + ' '
            if(cleared_line.endswith('.') or 
               cleared_line.endswith('?') or 
               cleared_line.endswith('!') or
               cleared_line.endswith("'") or
               cleared_line.endswith('"') or 
               cleared_line.startswith('CHAPTER')):
                file_lines.append(curr_line)
                curr_line = ''
    words = []
    for line in file_lines:
        # first get rid of punctuation
        first_normalization = normalize_line(line=line,regex_machine=punc_re_machine,replacment_str=" ")
        # then get rid of the quotations (` ' combo)
        second_normalization = normalize_line(line=first_normalization,regex_machine=quote_re_machine,replacment_str=r" \1 ")
        #then get rid of the quotations (" " combo)
        third_normalization = normalize_line(line=second_normalization,regex_machine=snd_quote_re_machine,replacment_str=r" \1 ")
        # now split the entire line
        norm_line_splitted = re.split(r'\s+',third_normalization)
        words.extend([x for x in norm_line_splitted if len(x) != 0])
    
    return words

def get_training_batches(tokens : list,word2idx : dict, window_size:int = 2):
    
    for i, word in enumerate(tokens):
        target_idx = word2idx[word]
        
        # boundaries of each word
        start = max(0, i-window_size)
        end = min(len(tokens),i+window_size+1)    

        context_indices = [
            word2idx[tokens[j]] for j in range(start, end) if j != i
        ]
        
        # prosljedjujemo sve ove parove u model, koji ce pozvati datu funkciju!
        yield target_idx, context_indices


if __name__ == '__main__':
    p = Path('/mnt2/jetbrains_hallucination/data/')
    filename = 'alice_in_wonderland.txt'
    words = preprocess_list(p,filename)
    word2idx, idx2word, vocab_size = build_vocab(words)
    