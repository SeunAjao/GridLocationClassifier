# for Python 2: use print only as a function
from __future__ import print_function

# ## Task 1
# 
# Read **`merged.csv`** into a pandas DataFrame and examine it.

# read merged.csv using a relative path
import pandas as pd
path = 'data/MergedGeo8US.csv'
merged = pd.read_csv(path, error_bad_lines=False, low_memory=False)


# examine the shape
merged.shape


# examine the first row
merged.head(1)


# examine the class distribution
merged.grid.value_counts().sort_index()


# define X and y using the original DataFrame
X = merged.text
y = merged.grid


# check that y contains 5 different classes
y.value_counts().sort_index()


# split X and y into training and testing sets; If train size is None, test size is automatically set to 0.25.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

# create document-term matrices using CountVectorizer
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

###########################################################################
#MULTINOMIAL NAIVE BAYES
# import and instantiate MultinomialNB
#from sklearn.naive_bayes import MultinomialNB
#nb = MultinomialNB()

# fit a Multinomial Naive Bayes model
#nb.fit(X_train_dtm, y_train)

# make class predictions
#y_pred_class = nb.predict(X_test_dtm)
##########################################################################

#DECISION TREES
#from sklearn import tree
#dtree = tree.DecisionTreeClassifier(random_state=0, max_depth=10)
#dtree.fit(X_train_dtm, y_train)
#y_pred_class = dtree.predict(X_test_dtm)

###########################################################################

#LOGISTIC REGERSSION
# import and instantiate a logistic regression model
#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()

# train the model using X_train_dtm
#logreg.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
#y_pred_class = logreg.predict(X_test_dtm)
###########################################################################

#RANDOM FOREST
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)
#clf.fit(X_train_dtm, y_train)
#y_pred_class = clf.predict(X_test_dtm)
###########################################################################

#NEURAL NETWORKS
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train_dtm, y_train)
y_pred_class = clf.predict(X_test_dtm)
###########################################################################

#GRADIENT BOOSTING
#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#clf.fit(X_train_dtm, y_train)
#y_pred_class = clf.predict(X_test_dtm)
#############################################################################

#SUPPORT VECTOR MACHINES
#from sklearn import svm
#clf = svm.SVC(kernel='linear', C =1.0)
#clf.fit(X_train_dtm, y_train)
#y_pred_class = clf.predict(X_test_dtm)

#############################################################################
# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# calculate the null accuracy
y_test.value_counts().head(1) / y_test.shape


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))
