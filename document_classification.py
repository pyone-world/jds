import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('trainingdata.txt')

df.columns = ['text']
df['label'] = df['text'].apply(lambda x: x.split(' ')[0])
df['text'] = df['text'].apply(lambda x: x.split(' ', 1)[1].rsplit(' ', 2)[0])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['text'])

# print(X_train_counts.shape)
# print(X_train_counts)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)

# print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, df['label'])
predicted = clf.predict(X_train_tfidf)

# print(sum(predicted == df['label']) / len(df))

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-6, n_iter_no_change=10, random_state=42))])

text_clf = text_clf.fit(df['text'][:5000], df['label'][:5000])

# print(sum(text_clf.predict(df['text']) == df['label']) / len(df))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-4, 1e-5, 1e-6, 1e-7)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(df['text'][:5000], df['label'][:5000])

# print(gs_clf.best_score_)
# print(gs_clf.best_params_)

# print(sum(text_clf.predict(df['text'][4000:]) == df['label'][4000:]) / len(df[4000:]))

# TEST CASE
test_case = open('document-classification-testcases/input/input03.txt').read().splitlines()
num_sents = int(test_case[0])

for s in test_case[1:]:
    print(text_clf.predict([s.strip()])[0])
