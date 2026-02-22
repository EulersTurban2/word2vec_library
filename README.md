# Word2Vec Implementation in Pure NumPy

## Project Overview
- Implemented **Word2Vec** (Skip-gram and CBOW) from scratch using **NumPy**.  
- Trained on a text corpus (e.g., *Alice in Wonderland*) to learn **dense word embeddings**.  
- No external ML frameworks were used.

## Preprocessing
- Normalized text: lowercasing, removed punctuation, handled single & double quotes.  
- Split corpus into **tokens** and built vocabulary with `word2idx` / `idx2word` mappings.  
- Applied **subsampling of frequent words** to reduce dominance of common words (like “the”, “and”).

## Model Architecture
- **Skip-gram**: Predicts surrounding context words given a target word.  
- **CBOW**: Predicts a target word from its surrounding context words.  
- **Embedding matrices**:  
  - `W_in` – input embeddings (words → vectors)  
  - `W_out` – output embeddings (vectors → words)  
- **Negative sampling**: Efficiently approximates softmax by sampling “negative” words for contrastive learning.

## Training Pipeline
- **Forward pass**: Compute scores between target and context (and negative) embeddings.  
- **Loss**: Binary cross-entropy using sigmoid activation.  
- **Gradients**: Computed for both input and output embeddings.  
- **Parameter updates**: Performed with stochastic gradient descent.  
- Split data into **training (80%)** and **validation (20%)** sets.

## Evaluation
- Computed **cosine similarity** between embeddings to find **nearest neighbors**.  
- Normalized embeddings **after training** for meaningful similarity comparisons.  
- Skip-gram captures semantic relationships better for rare words; CBOW averages context and may require more epochs.

## Saving / Loading
- Embedding matrices can be saved/loaded with **NumPy `.npz` files**.  
- Easy to resume training or use embeddings for downstream tasks.

## Lessons Learned
- Skip-gram more robust for small datasets / rare words.  
- CBOW requires careful gradient scaling due to averaging over context.  
- Subsampling and negative sampling dramatically improve training speed and embedding quality.