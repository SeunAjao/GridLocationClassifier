import numpy as np
import pandas as pd
import re
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from gensim.models import word2vec


MODEL_FILENAME = 'model.w2v'
SIMILARITY_THRESHOLD = 0.99


def setup():
    nltk.download('stopwords')
    global stop_words
    stop_words = set(stopwords.words('english'))    


def tokenize(tweet): 
    tweet = re.sub(r"http\S+", "", tweet)
    tokens = re.findall(r'\w+', tweet.lower())
    return [token for token in tokens if not token in stop_words]


def read_corpus(input_file):
    print("Reading corpus... ", end="")
    dataset = pd.read_csv(input_file, delimiter = '\t', usecols=["text", "label"], quoting = 3)
    dataset['text'] = dataset['text'].apply(tokenize)
    print("Done.")
    return dataset


def embed(corpus):
    print("Learning word2vec model... ", end="")
    model = word2vec.Word2Vec(corpus, min_count=1, size=300, window=7, workers=4)
    model.save(MODEL_FILENAME)
    print("Done.")
    return model


def dimreduce(model, corpus):
    print("Reducing dictionary... ", end="")

    to_replace = {}

    i = 0
    for word in model.wv.vocab:
        if i % 1000 == 0:
            print(i, "from", len(model.wv.vocab))
        synonim, similarity = model.wv.most_similar(word)[0]
        i += 1
        if similarity >= SIMILARITY_THRESHOLD and model.wv.vocab[synonim].count > model.wv.vocab[word].count:
            to_replace[word] = synonim
 
    for i in range(len(corpus)):
        corpus[i] = [to_replace[word] if word in to_replace else word for word in corpus[i]]

    print("Done.", len(model.wv.vocab), "->", len(model.wv.vocab) - len(to_replace))
    return corpus


def embedding_reduce(corpus, labels):
    model = embed(corpus)
    corpus = dimreduce(model, corpus)
    return pd.DataFrame({'text': corpus, 'label': labels })


def classify(corpus, labels):
    print("Learning logit model... ", end="")

    # transform into TF-IDF matrix
    corpus = [' '.join(tokens for tokens in tweet) for tweet in corpus]
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(corpus)
    y = labels

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Fit Logistic Regression model to the dataset
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_class = logreg.predict(X_test)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = logreg, X = X_train, y = y_train, cv = 5)
    accuracies.mean()
    NEWAccuracies = accuracies.mean()
    with open('NEWAccuracies_tweets.txt', 'w') as g:
        print(NEWAccuracies, file=g)
    
    # calculate accuracy of class predictions
    PreRecFM = metrics.classification_report(y_test, y_pred_class)
    with open('NEWPreRecFM_tweets.txt', 'w') as h:
        print(PreRecFM, file=h)

    print("Done.")
    return NEWAccuracies


def main():
    setup()
    
    input_file = sys.argv[1]
    dataset = read_corpus(input_file)
    
    # baseline learning
    old_accuracy = classify(dataset['text'], dataset['label'])
    
    # learning on reduced dataset
    dataset = embedding_reduce(dataset['text'], dataset['label'])
    new_accuracy = classify(dataset['text'], dataset['label'])

    print(old_accuracy, new_accuracy)


if __name__ == "__main__":
    main()