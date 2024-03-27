import pandas as pd
import ast

data = pd.read_csv('lemmatized_word_tokens.csv', index_col=0)
data.wordTokens = data.wordTokens.apply(ast.literal_eval)

data.to_pickle('lemmatized_word_tokens.pkl')
