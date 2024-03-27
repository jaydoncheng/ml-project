import pandas as pd

data = pd.read_csv('data.csv')
print('loaded data')

print(data.keys())
data['reviewText'].dropna(inplace=True)

"""**Preprocessing Step 6:** Tokenize the reviews. Two different tokenization methods are used, including word-based and sentence-based."""

import nltk
from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def custom_tokenize(text):
    try:
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        print('Error tokenizing text:', e)
        print('Text:', text)

print("Tokenizing reviews...")
data['wordTokens'] = data['reviewText'].apply(custom_tokenize)

# data['sentTokens'] = data['reviewText'].apply(sent_tokenize)

"""**Preprocessing Step 7:** Remove punctuation (except for `!` and `?`), special characters and numbers."""

def clean_tokens(tokens):
  cleaned_tokens = [re.sub(r'[^a-zA-Z\s!?]', '', token) for token in tokens]
  return [token for token in cleaned_tokens if token]

data['wordTokens'] = data['wordTokens'].apply(clean_tokens)

# data['sentTokens'] = data['sentTokens'].apply(clean_tokens)

"""**Preprocessing Step 8:** Remove stop words."""

from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def rm_stop_words(words):
 return [word for word in words if word not in stop_words]

data['wordTokens'] = data['wordTokens'].apply(rm_stop_words)

def rm_sent_stop_words(sentences):
  filtered_sentences = []

  for sentence in sentences:
    word_tokens = word_tokenize(sentence)
    filtered_sentences.append(' '.join([word for word in word_tokens if word not in stop_words]))

  return filtered_sentences

# data['sentTokens'] = data['sentTokens'].apply(rm_sent_stop_words)

"""**Preprocessing Step 9:** Lemmatize the reviews (reduce words to their base or root form, for instance 'crying' into 'cry')."""

# import spacy
#
# nlp = spacy.load("en_core_web_sm", exclude=['parser', 'ner'])
#
# def lemmatize_word_tokens(tokens):
#     lemmatized_tokens = []
#
#     for doc in nlp.pipe(tokens, n_process=-1):
#         lemmatized_tokens.append([token.lemma_ for token in doc])
#
#     return lemmatized_tokens
#
# data['wordTokens'] = lemmatize_word_tokens(data['wordTokens'].apply(lambda tokens: ' '.join(tokens)))
#
# def lemmatize_sent_tokens(tokens):
#     lemmatized_sentences = []
#
#     for token in tokens:
#       for doc in nlp.pipe(token, n_process=-1):
#         lemmatized_sentences.append(' '.join([token.lemma_ for token in doc]))
#
#     return lemmatized_sentences
#
#
# data['sentTokens'] = lemmatize_sent_tokens(data['sentTokens'].apply(lambda tokens: ' '.join(tokens)))
#
# with pd.option_context('display.max_colwidth', None):
#     print(data.iloc[7000])
#
# data.to_csv('preprocessed_data.csv', index=False)
