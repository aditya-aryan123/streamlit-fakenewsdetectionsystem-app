import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import pickle


fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')

fake_df['label'] = 0
real_df['label'] = 1

concat_df = pd.concat([fake_df, real_df])
concat_df = concat_df.sample(frac=1)

concat_df['content'] = concat_df['text'] + concat_df['title'] + concat_df['subject']
concat_df.drop(['date', 'title', 'text', 'subject'], axis=1, inplace=True)


def clean_text(text):
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [word for word in text if not word in stops and len(word) >= 3]
    text = " ".join(text)
    text = re.sub('[0-9]+', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('\.\.\.', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text


concat_df['content'] = concat_df['content'].apply(clean_text)

port_stem = PorterStemmer()


def stemming(content):
    return port_stem.stem(content)


concat_df['content'] = concat_df['content'].apply(stemming)

X = np.array(concat_df['content'])
y = np.array(concat_df['label'])

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
