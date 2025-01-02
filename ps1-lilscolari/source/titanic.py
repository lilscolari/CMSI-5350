"""
Author: Cameron Scolari
Date: 9/19/2024
Description: This is a Python file that analyzes different ways of training a DecisionTreeClassifier on 
Titanic data to best predict the survival outcome of passengers.
"""

# Use only the provided packages!
import math
import csv
from util import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object):
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

        y = None

        n,d = X.shape

        classes = list(self.probabilities_.keys())
        probabilities = list(self.probabilities_.values())

        y = np.random.choice(classes, size=n, p=probabilities)
    
        return y


######################################################################
# functions
######################################################################

def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
        test_size   -- float (between 0.0 and 1.0) or int,
                       if float, the proportion of the dataset to include in the test split
                       if int, the absolute number of test samples

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    training_errors = []
    test_errors = []

    for trial in range(ntrials):
        # Split data into train and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=trial)

        clf.fit(X_train, y_train)

        y_training_prediction = clf.predict(X_train)
        y_test_prediction = clf.predict(X_test)

        train_error = 1 - metrics.accuracy_score(y_train, y_training_prediction)
        test_error = 1 - metrics.accuracy_score(y_test, y_test_prediction)

        training_errors.append(train_error)
        test_errors.append(test_error)

    average_training_error = round(np.mean(training_errors), 3)
    average_test_error = round(np.mean(test_errors), 3)

    return average_training_error, average_test_error


def write_predictions(y_pred, filename, yname=None):
    """Write out predictions to csv file."""
    out = open(filename, 'w')
    f = csv.writer(out)
    if yname:
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


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
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)


    # train Decision Tree Classifier on data
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=20) # create Decision Tree Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y)       # Split data into train and test sets
    clf.fit(X_train, y_train)                                       # fit training data using the classifier
    y_pred = clf.predict(X_train)                                   # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    # note: uncomment out the following lines to output the Decision Tree graph
    
    """
    # save the classifier -- requires GraphViz and pydot
    import pydot
    from io import StringIO
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames,
                         class_names=['Died', 'Survived'])
    graph = pydot.graph_from_dot_data(str(dot_data.getvalue()))[0]
    graph.write_pdf('dtree.pdf')
    """

    print('Investigating various classifiers...')
    clf = MajorityVoteClassifier()
    majority_training_error, majority_test_error = error(clf, X, y)
    print(f"Majority Vote Classifier average training error: {majority_training_error} and average test error: {majority_test_error}")

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=20)
    decision_training_error, decision_test_error = error(clf, X, y)
    print(f"Decision Tree Classifier average training error: {decision_training_error} and average test error: {decision_test_error}")

    clf = RandomClassifier()
    random_training_error, random_test_error = error(clf, X, y)
    print(f"Random Classifier average training error: {random_training_error} and average test error: {random_test_error}")


    print('Investigating depths...')
    depths = list(range(1, 21))
    majority_test_errors = []
    random_test_errors = []
    tree_train_errors = []
    tree_test_errors = []

    clf = MajorityVoteClassifier()
    _, majority_test_error = error(clf, X, y)
    majority_test_errors.append(majority_test_error)
    majority_test_errors = [majority_test_errors] * 20

    clf = RandomClassifier()
    _, random_test_error = error(clf, X, y)
    random_test_errors.append(random_test_error)
    random_test_errors = [random_test_errors] * 20

    # Try DecisionTreeClassifier at different depths up to 20:
    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        tree_train_error, tree_test_error = error(clf, X, y)
        tree_train_errors.append(tree_train_error)
        tree_test_errors.append(tree_test_error)
    
    # Plot Classifiers vs. depth:
    plt.figure()
    plt.plot(depths, majority_test_errors, label="majority vote test")
    plt.plot(depths, random_test_errors, label="random test")
    plt.plot(depths, tree_train_errors, linestyle='dashed', label="decision tree train")
    plt.plot(depths, tree_test_errors, label="decision tree test")
    plt.xlabel("depth")
    plt.ylabel("error")
    plt.title("depth vs. error")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    # depth = 3 had the best test error of 0.198 


    print('Investigating training set sizes...')
    increments = np.arange(0.05, 1, 0.05)

    tree_train_errors = []
    tree_test_errors = []
    majority_test_errors = []
    random_test_errors = []

    # Try all Classifiers at different training data sizes:
    for increment in increments:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        tree_train_error, tree_test_error = error(clf, X, y, test_size=1-increment)
        tree_train_errors.append(tree_train_error)
        tree_test_errors.append(tree_test_error)

        clf = MajorityVoteClassifier()
        _, majority_test_error = error(clf, X, y, test_size=1-increment)
        majority_test_errors.append(majority_test_error)

        clf = RandomClassifier()
        _, random_test_error = error(clf, X, y, test_size=1-increment)
        random_test_errors.append(random_test_error)

    # Plot Classifiers by training data size:
    plt.figure()
    plt.plot(increments, majority_test_errors, label="majority vote test")
    plt.plot(increments, random_test_errors, label="random test")
    plt.plot(increments, tree_train_errors, linestyle='dashed', label="decision tree train")
    plt.plot(increments, tree_test_errors, label="decision tree test")
    plt.xlabel("amount of training data")
    plt.ylabel("error")
    plt.title("amount of training data vs. error")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


    # Building my own model:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize upper-bound model and error:
    best_clf = None
    min_error = 1

    # Update the data to combine "SibSp" and "Parch" columns:
    indices = []
    for index, name in enumerate(Xnames):
        if name == "SibSp" or name == "Parch":
            indices.append(index)

    new_X = []

    for i, row in enumerate(X):
        total = 0
        new_row = []
        for index, element in enumerate(row):
            if index not in indices:
                new_row.append(element)
            else:
                total += element

        new_row.append(total)
        new_X.append(new_row)

    # Looping through range(100) to test different random_state:
    for num in range(100):
        # Looping through 1-6 to see which max_features provides the lowest test error. max_features not known because I have clf updating automatically.
        for i in range(1, 6):
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=num, max_features=i)
            clf.fit(X_train, y_train)

            _, tree_test_error = error(clf, new_X, y)

            if tree_test_error < min_error:
                min_error = tree_test_error
                best_clf = clf
                # Print to see the lowest test error found through looping:
                print(tree_test_error)

    titanic_test = load_data('titanic_test.csv', header=1, predict_col=None)

    X_test = titanic_test.X

    new_X_test = []

    # Updating test data to matc the data I trained my model on:
    for i, row in enumerate(X_test):
        total = 0
        new_row = []
        for index, element in enumerate(row):
            if index not in indices:
                new_row.append(element)
            else:
                total += element

        new_row.append(total)
        new_X_test.append(new_row)


    y_pred = best_clf.predict(new_X_test)   # take the trained classifier and run it on the test data
    write_predictions(y_pred, '../data/cameron_titanic.csv', titanic.yname)

    print('Done')


if __name__ == '__main__':
    main()
