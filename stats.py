import pandas as pd

data = pd.read_pickle('lemmatized_word_tokens.pkl')


# Mean and max length of word tokens
print('mean length of word tokens: ', data.wordTokens.apply(len).mean())
print('max length of word tokens: ', data.wordTokens.apply(len).max())

# 90th percentile of length of word tokens
print('90th percentile of word tokens: ', data.wordTokens.apply(len).quantile(0.9))

# 95th percentile of length of word tokens
print('95th percentile of word tokens: ', data.wordTokens.apply(len).quantile(0.95))
