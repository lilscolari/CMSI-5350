"""
Author: Cameron Scolari
Date: 9/8/2024
Description: This is a Python file that implements some graphs to analyze Titanic passenger data. 
It also implements some Machine Learning techniques for predicting the passenger class.
"""

# Use only the provided packages!
import math
import csv
from util import *

from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier():
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier):

    def __init__(self):
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y):
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda val_count: val_count[1])
        self.prediction_ = majority_val
        return self

    def predict(self, X):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier):

    def __init__(self):
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y):
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        self.probabilities_ = dict()

        unique_classes, counts = np.unique(y, return_counts=True)

        total_target_classes = len(y)

        # Add respective probabilities to each unique class
        for c, count in zip(unique_classes, counts):
            self.probabilities_[c] = count / total_target_classes

        return self


    def predict(self, X, seed=1234):
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None:
            raise Exception('Classifier not initialized. Perform a fit first.')
        np.random.seed(seed)

        n,d = X.shape

        classes = list(self.probabilities_.keys())
        probabilities = list(self.probabilities_.values())

        y = np.random.choice(classes, size=n, p=probabilities)
    
        return y


######################################################################
# functions
######################################################################

def plot_histogram(X, y, Xname, yname):
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets:
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else:
        bins = 10
        align = 'mid'

    # plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show(block=True)


def plot_scatter(X, y, Xnames, yname):
    """
    Plots scatter plot of values in X grouped by y.

    Parameters
    --------------------
        X      -- numpy array of shape (n,2), feature values
        y      -- numpy array of shape (n,), target classes
        Xnames -- tuple of strings, names of features
        yname  -- string, name of target
    """

    targets = sorted(set(y))

    plt.figure()

    # Make scatter plot of individual passenger data grouped by whether they survived or not.
    for target in targets:
        plt.scatter(X[y == target, 0], X[y == target, 1], label=f'{yname}={target}', alpha=0.5)

    plt.title(f"{Xnames[0]} vs. {Xnames[1]}")
    plt.xlabel(Xnames[0])
    plt.ylabel(Xnames[1])
    plt.legend(loc='upper right')
    plt.show(block=True)


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data('titanic_train.csv', header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features


    #========================================
    # plot histograms of each feature
    print('Plotting...')
    for i in range(d):
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    # Find indeces of 'Age' and 'Fare'
    age_and_fare_indeces = []
    for i, name in enumerate(Xnames):
        if name == "Age":
            age_and_fare_indeces.append(i)
        if name == "Fare":
            age_and_fare_indeces.append(i)

    # Update X to include only the data belonging to either 'Age' or 'Fare'
    X = X[:, age_and_fare_indeces]

    plot_scatter(X, y, ["Age", "Fare"], yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    print('Classifying using Random...')

    # train Random classifier on data
    clf = RandomClassifier() # create Random classifier, which includes all model parameters
    clf.fit(X, y)            # fit training data using the classifier
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    print('Done')


if __name__ == '__main__':
    main()
