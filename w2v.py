from gensim.models import Word2Vec
import numpy as np
import os
import pandas as pd


model = None
data = pd.read_pickle('lemmatized_word_tokens.pkl')
print('loaded data')
print('clean empty word tokens')
data = data[data['wordTokens'].map(lambda d: len(d)) > 0]

dims = 300

if os.path.exists('w2v.model'):
    print('found existing w2v model, loading')
    model = Word2Vec.load('w2v.model')
else:

    print('training model')
    model = Word2Vec(sentences=data['wordTokens'].to_list(), vector_size=dims, window=5, min_count=1, workers=4, sg=0)
    print('vocab_size: ', len(model.wv.index_to_key))
    model.save('w2v.model')

max_seq_len = 300
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
        return np.zeros((dims,), dtype=np.float32)
    tokens = tokens[:max_seq_len]

    array = np.array([model.wv[word] for word in tokens if word in model.wv]) 
    print('c: ', category, end=' ')
    if len(array) < max_seq_len:
        zeros = np.zeros((max_seq_len-len(array), dims), dtype=np.float32)
        array = np.concatenate((array, zeros))
    print('a: ', array.shape)

    assert(array.shape == (max_seq_len, dims))
    assert(category in [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return array, category

wordVectors = []
wordMeanVectors = []
categories = []
categories_words = []
samples = 20000
for i, row in data[:samples].iterrows():
    try:
        print(i, end=' ')
        k, category = preprocess(row)
        wordVectors.append(k)
        wordMeanVectors.append(np.mean(k, axis=0))
        categories.append(category)
        categories_words.append(row.category)
    except Exception as e:
        print('error at row: ', i)
        print(e)
        continue

# wordVectors.to_pickle('word_vectors.pkl')
with open('samples/word_vectors.npy', 'wb') as f:
    np.save(f, wordVectors)
with open('samples/word_mean_vectors.npy', 'wb') as f:
    np.save(f, wordMeanVectors)
print('saved word_vectors.npy')
with open('samples/categories.npy', 'wb') as f:
    np.save(f, categories)
with open('samples/categories_words.npy', 'wb') as f:
    np.save(f, categories_words)
print('saved categories.npy')
