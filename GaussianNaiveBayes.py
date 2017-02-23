#!/usr/bin/env python3

"""
CS 445/545
Machine Learning
Winter 2017
Homework 4: Naive Bayes Classification and Logistic Regression
Due Thursday Feb. 23, 2017

In this homework you will use Gaussian Naïve Bayes and Logistic Regression to classify the Spambase data from
the UCI ML repository (the same dataset you worked with in Homework 3).

Part I: Classification with Naïve Bayes
1. Create training and test set: Split the data into a training and test set. Each of these should have
   about 2,300 instances, and each should have about 40% spam, 60% not-spam, to reflect the statistics of the
   full data set.

2. Create probabilistic model. (Write your own code to do this.)
    • Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in the training data. As
      described in part 1, P(1) should be about 0.4.
    • For each of the 57 features, compute the mean and standard deviation in the training set of the values
      given each class.

3. Run Naïve Bayes on the test data. (Write your own code to do this.)
    • Use the Gaussian Naïve Bayes algorithm to classify the instances in your test set

In your report, include a short description of what you did, and your results: the accuracy, precision,
and recall on the test set, as well as a confusion matrix for the test set. Write a few sentences
describing your results, and answer these questions: How did Naïve Bayes do compared with your SVM from
Homework 3? Do you think the attributes here are independent, as assumed by Naïve Bayes? Does Naïve Bayes
do well on this problem in spite of the independence assumption? Speculate on other reasons Naïve Bayes
might do well or poorly on this problem.


Part II: Classification with Logistic Regression

Find a library that performs logistic regression (for example, scikit_learn in Python). Use this library to
train a logistic regression model with your training set from Part I, and to test this model with your test
set from Part I. Give the following in your report:

    • Describe what library you used and what parameter values you used in running logistic regression.
    • Give the accuracy, precision, and recall of your learned model on the test set, as well as a confusion
      matrix.
    • Write a few sentences comparing the results of logistic regression to those you obtained from Naïve Bayes
      and from your SVM from Homework 3.
"""

__author__ = "Mike Lane"
__copyright__ = "Copyright 2017, Michael Lane"
__license__ = "MIT"
__email__ = "mikelane@gmail.com"

import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics


def sum_of_log_probs(P_class, x, mus, sigmas):
    """
    Return the sum of the log of probabilities.
    """
    return np.log(P_class) + np.log((1 / (np.sqrt(2 * np.pi) * sigmas)) * np.exp(-((x - mus) ** 2 / (2 * sigmas ** 2)))).sum()


def predict(x):
    """
    Predict the class of a given instance.
    :param x: A numpy array or pandas DataFrame.
    :return: 0 or 1
    """
    p_spam = sum_of_log_probs(prob_spam_train, x, train_spam_col_means, train_spam_col_stds)
    p_ham = sum_of_log_probs(prob_ham_train, x, train_ham_col_means, train_ham_col_stds)
    return 1 if p_spam > p_ham else 0

print(__doc__)

# Get the data setting the index to be the class
data = pd.read_csv('~/data/spam.data/spambase.data', header=None, index_col=57)

# Shuffle and split the data into a testing and training set.
X_train, X_test = sklearn.model_selection.train_test_split(data,
                                                           test_size=0.5,
                                                           random_state=np.random.RandomState())

# Calculate the prior probabilities
prob_spam_train = len(X_train.loc[1].index) / len(X_train.index)
prob_ham_train = len(X_train.loc[0].index) / len(X_train.index)

# Pull out the mean and standard deviation (this step doesn't meet the "code it yourself" requirement)
train_spam_col_means = X_train.loc[1].describe().loc['mean']
train_spam_col_stds = X_train.loc[1].describe().loc['std']
train_ham_col_means = X_train.loc[0].describe().loc['mean']
train_ham_col_stds = X_train.loc[0].describe().loc['std']

# Apply the predict function to each row of the testing data
NB_y_pred = X_test.apply(predict, axis=1)

# Calculate the confusion matrix, accuracy, precision and recall
NB_confusion_matrix = sklearn.metrics.confusion_matrix(y_true=X_test.index.values, y_pred=NB_y_pred)
NB_accuracy = sklearn.metrics.accuracy_score(y_true=X_test.index.values, y_pred=NB_y_pred)
NB_precision = sklearn.metrics.precision_score(y_true=X_test.index.values, y_pred=NB_y_pred)
NB_recall = sklearn.metrics.recall_score(y_true=X_test.index.values, y_pred=NB_y_pred)
print('\nGaussian Naive Bayes classifier:\n'
      '   accuracy: {:.5f}\n'
      '  precision: {:.5f}\n'
      '     recall: {:.5f}\n'
      'conf matrix:\n'
      '{}'.format(NB_accuracy, NB_precision, NB_recall, NB_confusion_matrix))

print('\n', '-' * 50, '\n')

# Part 2, Use a logistic regressor fuction from a library and classify the spam data.
logistic_regressor = sklearn.linear_model.LogisticRegression().fit(X_train, X_train.index.values)
LR_y_pred = logistic_regressor.predict(X_test)

# Calculate the confusion matrix, accuracy, precision, and recall
LR_confusion_matrix = sklearn.metrics.confusion_matrix(y_true=X_test.index.values, y_pred=LR_y_pred)
LR_accuracy = sklearn.metrics.accuracy_score(y_true=X_test.index.values, y_pred=LR_y_pred)
LR_precision = sklearn.metrics.precision_score(y_true=X_test.index.values, y_pred=LR_y_pred)
LR_recall = sklearn.metrics.recall_score(y_true=X_test.index.values, y_pred=LR_y_pred)

print('Logistic Regression classifier:\n'
      '   accuracy: {:.4f}\n'
      '  precision: {:.4f}\n'
      '     recall: {:.4f}\n'
      'conf matrix:\n'
      '{}\n'.format(LR_accuracy, LR_precision, LR_recall, LR_confusion_matrix))
