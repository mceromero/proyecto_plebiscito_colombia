# -*- coding: utf-8 -*-

# import modules
import pandas as pd
import numpy as np
import codecs
import json

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, \
	RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
	precision_score, recall_score, f1_score, \
	confusion_matrix, classification_report

# joblib -> saving model
import joblib

# stop assets
from assets import stopwords
from assets import stopsigns


# Utils

def load_stop_assets(vocabulary=[]):
	'''
	'''
	obj = {}
	assets = stopwords
	obj['words'] = [i for i in assets if i not in vocabulary]
	
	# load stop signs
	obj['punctuation'] = stopsigns
	
	return obj


def stop_words(split_document, stop_assets):
    '''
    '''
    return [
		w for w in split_document if w not in stop_assets['words']
	]


def clean_document(split_document, stop_assets):
    '''
    '''
    for p in stop_assets['punctuation']:
        split_document = [w.replace(p, ' ').strip() for w in split_document]
    
    # clean document
    document = ' '.join(split_document).strip().lower()
    
    # stop words
    split_document = document.split()
    return stop_words(split_document, stop_assets)


def bag_of_words(document, stop_assets):
    '''
    '''
    split_document = document.split()
    return clean_document(split_document, stop_assets)


def get_polarity_score(tokens, vocabulary):
    '''
    '''
    scores = []
    for w in tokens:
        if w in vocabulary.keys():
            s = vocabulary[w]
            scores.append(s)
    
    result = 0.5
    if len(scores) > 0:
        result = sum(scores) / len(scores)
    
    return result

# Load vocabulary
root = './assets/vocab_dict_corpus.json'
with codecs.open(root, encoding='latin-1') as f:
    vocabulary = json.load(f)
    f.close()


# load csv data

'''
Para crear un modelo de predicción, haciendo uso del modelo de
BernoulliNB, es necesario cargar en esta línea de código una base
de datos prreviamente clasificada (e.g., Negativo, Positivo)

Warning:

De no contar con una base que tenag estas características, el
código mostrará error en la carga de un archivo.

Este script considera que la base de datos incluye dos campos
específicos

- Content (Texto)
- Polarity (Clasificación del contenido [e.g., N o P])
'''

path = ''
data = pd.read_csv(path, encoding='utf-8', low_memory=False)


# bag of words and vocabulary
stop_assets = load_stop_assets(vocabulary=list(vocabulary.keys()))

data['Tokens'] = [
	bag_of_words(doc, stop_assets) for doc in data['Content']
]

data['Tokens_sentence'] = [
	' '.join(token) for token in data['Tokens']
]

vocab = list(vocabulary.keys())
stopw = stop_assets['words']


# Train set and Test set
X = data['Content'].copy()
y = data['Polarity'].copy()

stratified_split = StratifiedShuffleSplit(
	n_splits=1, test_size=0.20, random_state=seed
)

for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# reset index
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Create Pipeline
pipeline = Pipeline([
    ('vector', CountVectorizer(stop_words=stopw)),
    ('tfidf', TfidfTransformer()),
    ('classifier', BernoulliNB(alpha=0.1))
])

# Training model
parameters = {
    'vector__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
    'vector__min_df': (1, 2, 3, 5),
    'tfidf__use_idf': (False, True),
    'classifier__alpha': (1.0, 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.001)
}

grid = GridSearchCV(pipeline, cv=15, param_grid=parameters, verbose=0)
grid.fit(X_train, y_train)


y_preds = grid.predict(X_test)
y_test_array = np.array(y_test)

# debug metrics
print (f'accuracy score -> {accuracy_score(y_test, y_preds)}')
p_score_micro = precision_score(y_test_array, y_preds, average='micro')
print (f'precision score micro -> {p_score_micro}')
p_score_macro = precision_score(y_test_array, y_preds, average='macro')
print (f'precision score macro -> {p_score_macro}')
recall_micro = recall_score(y_test_array, y_preds, average='micro')
print (f'recall score micro -> {recall_micro}')
recall_macro = recall_score(y_test_array, y_preds, average='macro')
print (f'recall score macro -> {recall_macro}')
f1_score_micro = f1_score(y_test_array, y_preds, average='micro')
print (f'f1 score micro -> {f1_score_micro}')
f1_score_macro = f1_score(y_test_array, y_preds, average='macro')
print (f'f1 score macro -> {f1_score_macro}')
matthews_corrcoef_ = matthews_corrcoef(y_test_array, y_preds)
print (f'matthews_corrcoef -> {matthews_corrcoef_}')

# accuracy score
print (f'accuracy score\n{confusion_matrix(y_test, y_preds)}')
print (f'accuracy score\n{classification_report(y_test, y_preds)}')

# saving model
root = 'plebiscito_BernoulliNB.best.pkl'
joblib.dump(grid, root)
