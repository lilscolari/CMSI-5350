"""
Author: Cameron Scolari
Date: 10/10/24
Description: Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score

######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs):
    """
    Determine classifier performance across multiple trials using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one model_selection.KFold object

    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """

    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))

    # running multiple trials of cross validation:
    for trial, kf in enumerate(kfs):
        all_scores = cv_performance_one_trial(clf, train_data, kf)
        scores[trial] = all_scores
    return scores


def cv_performance_one_trial(clf, train_data, kf):
    """
    Compute classifier performance across multiple folds using cross-validation

    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold

    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """

    scores = np.zeros(kf.n_splits)

    # running one trial of cross validation:
    for fold, (train_index, test_index) in enumerate(kf.split(train_data.X)):
        X_train, X_val = train_data.X[train_index], train_data.X[test_index]
        y_train, y_val = train_data.y[train_index], train_data.y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        scores[fold] = accuracy
    return scores


######################################################################
# main
######################################################################

def main():
    np.random.seed(1234)

    #========================================
    # load data
    train_data = load_data('phoneme_train.csv')

    # getting training error for perceptron classifier:
    clf = Perceptron(fit_intercept=True)
    clf.fit(train_data.X, train_data.y)
    perceptron_y_pred = clf.predict(train_data.X)
    perceptron_accuracy = accuracy_score(train_data.y, perceptron_y_pred)
    perceptron_error = 1 - perceptron_accuracy

    # getting training error for logistic regression classifier:
    clf = LogisticRegression(C=1e10, solver='liblinear', fit_intercept=True)
    clf.fit(train_data.X, train_data.y)
    logistic_y_pred = clf.predict(train_data.X)
    logistic_accuracy = accuracy_score(train_data.y, logistic_y_pred)
    logistic_error = 1 - logistic_accuracy

    # a perceptron training error of 0 means the data is linearly separable:
    print(f'Perceptron training error: {perceptron_error}')
    print(f'Logistic Regression training error: {logistic_error}')

    # Helper method:
    def mean_and_stdev(scores):
        print(f'Average accuracy across trials: {np.round(scores.mean(axis=1).mean(), 3)}')
        print(f'Average standard deviation across trials: {np.round(scores.std(axis=1).mean(), 3)}')

    n_trials = 10
    n_splits = 10

    kfs = [model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(1234)) for _ in range(n_trials)]

    print("No preprocessing:")

    clf = Perceptron(fit_intercept=True)
    scores = cv_performance(clf, train_data, kfs)
    print("Perceptron:")
    mean_and_stdev(scores)

    clf = LogisticRegression(C=1e10, solver='liblinear', fit_intercept=True)
    scores = cv_performance(clf, train_data, kfs)
    print("Logistic Regression (no regularization):")   
    mean_and_stdev(scores)

    clf = LogisticRegression(C=1, solver='liblinear', fit_intercept=True)
    scores = cv_performance(clf, train_data, kfs)
    print("Logistic Regression (regularization):")
    mean_and_stdev(scores)

    print("Preprocessing:")

    scaler = preprocessing.StandardScaler()

    standardized_X = scaler.fit_transform(train_data.X)

    standardized_train_data = Data(standardized_X, train_data.y)

    clf = Perceptron(fit_intercept=True)
    scores = cv_performance(clf, standardized_train_data, kfs)
    print("Perceptron:")
    mean_and_stdev(scores)

    clf = LogisticRegression(C=1e10, solver='liblinear', fit_intercept=True)
    scores = cv_performance(clf, standardized_train_data, kfs)
    print("Logistic Regression (no regularization):")
    mean_and_stdev(scores)

    clf = LogisticRegression(C=1, solver='liblinear', fit_intercept=True)
    scores = cv_performance(clf, standardized_train_data, kfs)
    print("Logistic Regression (regularization):")
    mean_and_stdev(scores)

if __name__ == '__main__':
    main()
