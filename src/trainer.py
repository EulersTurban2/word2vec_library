import os
path_splitted = __file__.split(os.sep)
master_dir = ''
for entry in path_splitted:
    if entry == 'src':
        break    
    master_dir = master_dir + entry + os.sep
os.chdir(master_dir)

import numpy as np
import src.utils as ut

from src.sal import SAL
from pathlib import Path
from src.utils import NegativeSampler
from src.model import SkipGram_Model, CBOW_Model
from src.preprocessing import get_training_batches, preprocess_list


def train_step(model:SkipGram_Model|CBOW_Model, target_idx:int, context_idx:int|list[int], negative_indices:list[int], lr: float = 0.01):
    # We do a forward pass
    v_w, u_vectors, scores, all_indices = model.forward(target_idx=target_idx, context_idx=context_idx, negative_indices=negative_indices)

    # We now make a prediction
    predictions = ut.sigmoid(scores) # (1, 1 + len(neg_indices))

    # the target value of the rest
    labels = np.zeros(len(all_indices))
    labels[0] = 1.0

    # We calculate the errors
    errors = predictions - labels # (1, 1 + len(neg_indices))

    grad_u_vector = np.outer(v_w,errors)
    grad_v_w = np.dot(u_vectors,errors) 
    if isinstance(model,SkipGram_Model):
        # Now let's calculate the gradients
        grad_v_w = np.dot(u_vectors,errors)        

        # we update the embedding and contex matrices!
        model.W_in[target_idx] -= lr * grad_v_w
    else:
        # update all context embeddings proportionally
        for ctx_idx in context_idx:
            model.W_in[ctx_idx] -= lr * grad_v_w / len(context_idx)

    for i, idx in enumerate(all_indices):
        model.W_out[:, idx] -=  lr * grad_u_vector[:, i]


    loss = ut.cross_entropy_loss(labels=labels,predictions=predictions,error_margin=1e-9)
    return loss

def train_and_validate(tokens: list, word2idx: dict, model:SkipGram_Model|CBOW_Model, num_epochs = 5, window_size = 2,  n_samples = 5, lr=0.01, train_test_split:float = 0.8):
    
    ## we take the first (train_test_split)% of tokens for training

    tokens_size = len(tokens) 
    train_size = int(train_test_split*tokens_size)

    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]

    sampler = NegativeSampler(tokens,n_samples=n_samples)

    for epoch in range(num_epochs):
        if isinstance(model,SkipGram_Model):
            # training
            train_batches = get_training_batches(train_tokens, word2idx, window_size)
            # validation
            val_batches = get_training_batches(val_tokens, word2idx, window_size)
            
            total_loss = 0.0
            step_count = 0

            for target_id, context_list in train_batches:
                for context_id in context_list:

                    neg_indices = sampler.sample(target_idx=target_id,
                                                context_idx=context_id,
                                                word2idx=word2idx)

                    loss = train_step(model=model,
                                    target_idx=target_id,
                                    context_idx=context_id,
                                    negative_indices=neg_indices,
                                    lr=lr)

                    total_loss += loss
                    step_count += 1

            avg_loss = total_loss / step_count if step_count > 0 else 0.0
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

            # Validation loop
            val_loss = 0.0
            val_steps = 0
            for target_idx, context_indices in val_batches:
                # forward pass only, no updates
                _, _, scores, all_indices = model.forward(target_idx, context_indices[0], negative_indices=[])  # optionally skip negatives
                predictions = ut.sigmoid(scores)
                labels = np.zeros(len(all_indices))
                labels[0] = 1.0
                val_loss += ut.cross_entropy_loss(labels=labels, predictions=predictions,error_margin=1e-9)
                val_steps += 1

            avg_val_loss = val_loss / val_steps
            print(f"Validation Loss: {avg_val_loss:.6f}")
        else:
            # training
            train_batches = get_training_batches(train_tokens, word2idx, window_size)
            # validation
            val_batches = get_training_batches(val_tokens, word2idx, window_size)
            
            total_loss = 0.0
            step_count = 0

            for target_id, context_list in train_batches:

                neg_indices = sampler.sample(target_idx=target_id,
                                            context_idx=context_list[0],
                                            word2idx=word2idx)

                loss = train_step(model=model,
                                target_idx=target_id,
                                context_idx=context_list,
                                negative_indices=neg_indices,
                                lr=lr)

                total_loss += loss
                step_count += 1

            avg_loss = total_loss / step_count if step_count > 0 else 0.0
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

            # Validation loop
            val_loss = 0.0
            val_steps = 0
            for target_idx, context_indices in val_batches:
                # forward pass only, no updates
                _, _, scores, all_indices = model.forward(target_idx, context_indices, negative_indices=[])  # optionally skip negatives
                predictions = ut.sigmoid(scores)
                labels = np.zeros(len(all_indices))
                labels[0] = 1.0
                val_loss += ut.cross_entropy_loss(labels=labels, predictions=predictions,error_margin=1e-9)
                val_steps += 1

            avg_val_loss = val_loss / val_steps
            print(f"Validation Loss: {avg_val_loss:.6f}")

if __name__ == '__main__':
    p = Path('/home/gordan/Desktop/jetbrains/word2vec_library/data')
    filename = 'alice_in_wonderland.txt'
    words = preprocess_list(p,filename)
    word2idx, idx2word, vocab_size = ut.build_vocab(tokens=words)
    freqs = ut.compute_word_frequencies(tokens=words)
    subsampled_words = ut.subsample_tokens(tokens=words,freqs=freqs,t=1e-2)
    model = CBOW_Model(vocab_size=vocab_size)
    train_and_validate(tokens=subsampled_words,word2idx=word2idx,model=model)
    sal = SAL(model=model,word2idx=word2idx,idx2word=idx2word)
    sal.save('/home/gordan/Desktop/jetbrains/word2vec_library/models')