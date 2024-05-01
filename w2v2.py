from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack


data = pd.read_pickle('lemmatized_word_tokens.pkl')
sentences = data['wordTokens'].tolist()

w2v_custom = Word2Vec(sentences, vector_size=300, window=5, min_count=2, workers=4, sg=0)
i = 0

def get_avg_feature_vecs(words, model):
    global i

    model = model.wv if hasattr(model, 'wv') else model
    vec = np.zeros(model.vector_size, dtype="float32")
    model_set = set(model.index_to_key)
    num_words = 0

    for word in words:
      if word in model_set:
        num_words = num_words + 1
        vec = np.add(vec, model[word])

    print(i, end='\r')
    i += 1
    return np.divide(vec, num_words)

i = 0
data['word2vec_custom_avg'] = data['wordTokens'].apply(lambda words : get_avg_feature_vecs(words, w2v_custom))
print('done custom')
# i = 0
# data['word2vec_google_avg'] = data['wordTokens'].apply(lambda words : get_avg_feature_vecs(words, w2v_google))
# print('done google')

def get_padded_feature_vec_list(words, model, max_len):
    global i
    model = model.wv if hasattr(model, 'wv') else model
    model_set = set(model.index_to_key)
    vecs = []

    for word in words:
      if word in model_set:
        vecs.append(model[word])

    if len(vecs) > 0:
        vecs_sparse = csr_matrix(vecs, dtype=np.float32)
    else:
        vecs_sparse = csr_matrix((1, model.vector_size), dtype=np.float32)

    i += 1
    print(i, end='\r')
    if len(vecs) < max_len:
        missing_rows = max_len - len(vecs)
        pad = csr_matrix((missing_rows, model.vector_size), dtype=np.float32)
        result = vstack([vecs_sparse, pad])
    else:
        result = vecs_sparse

    return result if result.shape[0] == max_len else result[:max_len]

max_token_len = 300

data.loc[data['wordTokens'].apply(len) == max_token_len].to_csv("t.csv", index=False)
i = 0
data['word2vec_custom_list'] = data['wordTokens'].apply(lambda words : get_padded_feature_vec_list(words, w2v_custom, max_token_len))
print('done custom')
# i = 0
# data['word2vec_google_list'] = data['wordTokens'].apply(lambda words : get_padded_feature_vec_list(words, w2v_google, max_token_len))
# print('done custom')

data.to_pickle('word_vectors.pkl')
