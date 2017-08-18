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
COSINE_THRESHOLD = 0.99
JACCARD_THRESHOLD = 0.95


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
    return dataset['text'], dataset['label']


def embed(corpus):
    print("Learning word2vec model... ", end="")
    model = word2vec.Word2Vec(corpus, min_count=1, size=300, window=7, workers=4)
    model.save(MODEL_FILENAME)
    print("Done.")
    return model


def jaccard(x, y):
    return sum(min(x[i], y[i]) for i in range(len(x))) / sum(max(x[i], y[i]) for i in range(len(x)))


def normalize(model):
    result = {}
    for word in model.wv.vocab:
        result[word] = model.wv[word] + abs(min(model.wv[word]))
    return result


def dimreduce(model, corpus):
    print("Reducing dictionary... ", end="")

    to_replace = {}

    # For generalized Jaccard
    normalized_wv = normalize(model)

    i = 0
    for word in model.wv.vocab:
        if i % 1000 == 0:
            print(i, "from", len(model.wv.vocab))
        i += 1

        synonim, cosin_sim = model.wv.most_similar(word)[0]
        jaccard_sim = jaccard(normalized_wv[word], normalized_wv[synonim])
        
        if cosin_sim >= COSINE_THRESHOLD and jaccard_sim >= JACCARD_THRESHOLD and model.wv.vocab[synonim].count > model.wv.vocab[word].count:
            to_replace[word] = synonim
 
    for i in range(len(corpus)):
        corpus[i] = [to_replace[word] if word in to_replace else word for word in corpus[i]]

    # save fixed words
    with open("dictionary.txt", 'w') as g:
        print("\n".join("{}\t{}".format(k, v) for k, v in to_replace.items()), file=g)

    print("Done.", len(model.wv.vocab), "->", len(model.wv.vocab) - len(to_replace))
    return corpus


def embedding_reduce(corpus, labels):
    print("Embedding... ")
    model = embed(corpus)
    corpus = dimreduce(model, corpus)
    print("Done.")
    return corpus, labels


def classify(corpus, labels, output_filenames):
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
    with open(output_filenames[0], 'w') as g:
        print(NEWAccuracies, file=g)
    
    # calculate accuracy of class predictions
    PreRecFM = metrics.classification_report(y_test, y_pred_class)
    with open(output_filenames[1], 'w') as h:
        print(PreRecFM, file=h)

    print("Done.")
    return NEWAccuracies


def main():
    setup()
    
    input_file = sys.argv[1]
    corpus, labels = read_corpus(input_file)
    
    # baseline learning
    old_accuracy = classify(corpus, labels, 
                            ['old_Accuracies_tweets.txt', 'old_PreRecFM_tweets.txt'])
    
    # learning on reduced dataset
    corpus, labels = embedding_reduce(corpus, labels)
    new_accuracy = classify(corpus, labels,
                            ['new_Accuracies_tweets.txt', 'new_PreRecFM_tweets.txt'])

    print(old_accuracy, new_accuracy)


if __name__ == "__main__":
    main()