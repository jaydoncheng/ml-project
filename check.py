import pandas as pd

data = pd.read_pickle('word_vectors.pkl')

i = 0
try:
    for x in data['wordVectors']:
        if len(x) != 200:
            print(len(x))
            print(x)
            break
        i += 1
except:
    print('error')
    print(data.iloc[i])
