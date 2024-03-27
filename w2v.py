from gensim.models import Word2Vec
import numpy as np
import os
import pandas as pd


model = None
data = pd.read_pickle('lemmatized_word_tokens.pkl')
print('loaded data')
print('clean empty word tokens')
data = data[data['wordTokens'].map(lambda d: len(d)) > 0]
if os.path.exists('w2v.model'):
    print('found existing w2v model, loading')
    model = Word2Vec.load('w2v.model')
else:

    print('training model')
    model = Word2Vec(sentences=np.array(data['wordTokens']), vector_size=100, window=5, min_count=1, workers=4)
    print('vocab_size: ', len(model.wv.index_to_key))
    model.save('w2v.model')

max_seq_len = 100
def preprocess(row):

    category = row.category
    if category == 'positive':
        category = [1, 0, 0]
    elif category == 'negative':
        category = [0, 0, 1]
    elif category == 'neutral':
        category = [0, 1, 0]
    tokens = row.wordTokens

    if len(tokens) == 0:
        return np.zeros((100,), dtype=np.float32)
    tokens = tokens[:max_seq_len]

    array = np.array([model.wv[word] for word in tokens if word in model.wv]) 
    print('c: ', category, end=' ')
    if len(array) < max_seq_len:
        zeros = np.zeros((max_seq_len-len(array), 100), dtype=np.float32)
        array = np.concatenate((array, zeros))
    print('a: ', array.shape)

    assert(array.shape == (max_seq_len, 100))
    assert(category in [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return array, category

wordVectors = []
categories = []
for i, row in data.iterrows():
    try:
        print(i, end=' ')
        k, category = preprocess(row)
        wordVectors.append(k)
        categories.append(category)
    except Exception as e:
        print('error at row: ', i)
        print(e)
        continue

# wordVectors.to_pickle('word_vectors.pkl')
with open('word_vectors.npy', 'wb') as f:
    np.save(f, wordVectors)
print('saved word_vectors.npy')
with open('categories.npy', 'wb') as f:
    np.save(f, categories)
print('saved categories.npy')
