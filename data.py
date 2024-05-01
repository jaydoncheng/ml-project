import pandas as pd
import numpy as np

data = pd.read_pickle('vectorized_tokens.pkl')

wordVectors = data['word2vec_custom_list']

# Reshape the word vectors so the dataframe is a 3d tensor
wordVectors = np.array(wordVectors.to_list())

with open('word_vectors.npy', 'wb') as f:
    np.save(f, wordVectors, allow_pickle=True)
