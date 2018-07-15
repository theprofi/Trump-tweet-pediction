from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.transform(['Something second completely new.']).toarray())
print(vectorizer.transform(['Snew.']).toarray())


lll = []
lll.append("2") if 2>1 else lll.append("noo")
print(lll)