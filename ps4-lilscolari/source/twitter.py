"""
Author: Cameron Scolari
Date: 10/29/2024
Description: This file contains a bunch of functions that assist us in using SVMs.
"""

import numpy as np

from string import punctuation

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle


######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    with open(infile, 'r') as fid:
        file_string = fid.read()
        index = 0
        for word in extract_words(file_string):
            if word in word_list.keys():
                continue
            word_list[word] = index
            index += 1
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'r') as fid:
        for line_number, line in enumerate(fid):
            for word in extract_words(line):
                if word in word_list.keys():
                    feature_matrix[line_number, word_list[word]] = 1
                else:
                    pass
    return feature_matrix


def test_extract_dictionary(dictionary):
    err = 'extract_dictionary implementation incorrect'

    assert len(dictionary) == 1811

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0, 100, 10)]
    assert exp == act


def test_extract_feature_vectors(X):
    err = 'extract_features_vectors implementation incorrect'

    assert X.shape == (630, 1811)

    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all()


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric='accuracy'):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1

    match metric:
        case "accuracy":
            return metrics.accuracy_score(y_true, y_label)
        case "f1_score":
            return metrics.f1_score(y_true, y_label)
        case "auroc":
            return metrics.roc_auc_score(y_true, y_pred)
        case "precision":
            return metrics.precision_score(y_true, y_label)
        case "sensitivity":
            return metrics.recall_score(y_true, y_label)
        case "specificity":
            true_negative, false_positive, false_negative, true_positive = metrics.confusion_matrix(y_true, y_label).ravel()
            specificity = true_negative / (true_negative + false_positive)
            return specificity

def test_performance():
    """Ensures performance scores are within epsilon of correct scores."""

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics):
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])

def cv_performance(clf, X, y, kf, metric='accuracy'):
    """
    Splits the data, X and y, into k folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make "continuous-valued" predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()

def select_param_linear(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that maximizes the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)

    best_C = None
    best_score = -np.inf

    for C in C_range:
        clf = SVC(C=C, kernel='linear')

        average_score = cv_performance(clf, X, y, kf, metric)

        if average_score > best_score:
            best_score = average_score
            best_C = C

    return best_C

def select_param_rbf(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

    best_C = None
    best_gamma = None
    best_score = -np.inf

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-5, 2)

    for C in C_range:
        for gamma in gamma_range:
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            average_score = cv_performance(clf, X, y, kf, metric)

            if average_score > best_score:
                best_score = average_score
                best_C = C
                best_gamma = gamma

    return best_gamma, best_C

def performance_CI(clf, X, y, metric='accuracy'):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower, upper -- tuple of floats, confidence interval
    """

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)

    n = len(y)
    t = 1000
    bootstrap_scores = []

    for _ in range(t):
        bootstrap_indices = np.random.randint(0, n, size=n)

        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        y_pred_bootstrap = clf.decision_function(X_bootstrap)
        bootstrap_score = performance(y_bootstrap, y_pred_bootstrap, metric)
        
        bootstrap_scores.append(bootstrap_score)

    bootstrap_scores = np.sort(bootstrap_scores)

    lower_bound = bootstrap_scores[24]
    upper_bound = bootstrap_scores[974]

    return score, lower_bound, upper_bound


######################################################################
# main
######################################################################

def main():
    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    test_extract_feature_vectors(X)
    y = read_vector_file('../data/labels.txt')

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    test_performance()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = {metric: [] for metric in metric_list}

    C_range = 10.0 ** np.arange(-3, 3)

    for metric in metric_list:
        for C in C_range:
            clf = SVC(C=C, kernel='linear')
            score = cv_performance(clf, X_train, y_train, skf, metric)
            results[metric].append(score)

    print(results)

    results = {metric: [] for metric in metric_list}

    for metric in metric_list:
        results[metric] = select_param_rbf(X, y, skf, metric)
    
    print(results)

    for metric, params in results.items():
        clf = SVC(kernel='rbf', C=params[1], gamma=params[0])
        average_score = cv_performance(clf, X, y, skf, metric)
        print(f"{metric} score for best hyperparameters{params} is: {average_score}")

    clf = SVC(C=1, kernel='linear')
    clf.fit(X_train, y_train)

    for metric in metric_list:
        score, lower, upper = performance_CI(clf, X_test, y_test, metric)
        print(f"{metric}:\nscore={score}\n95% CI=[{lower}, {upper}]")

    top_10_indices = np.argsort(clf.coef_[0])[-10:][::-1]
    bottom_10_indices = np.argsort(clf.coef_[0])[:10]
    
    reversed_dictionary = {index: word for word, index in dictionary.items()}

    top_10_words = [reversed_dictionary[index] for index in top_10_indices]
    bottom_10_words = [reversed_dictionary[index] for index in bottom_10_indices]

    print(f"The top 10 features are:\n{top_10_words}")
    print(f"The bottom 10 features are:\n{bottom_10_words}")

    positive_words = {'love', 'great', 'happy', 'must', 'funny', 'good', 'excellent', 'liked', 'awesome'}
    negative_words = {'hate', 'bad', 'sad', 'not', 'shit', 'hated', 'dissapointing', 'boring'}

    for word in positive_words:
        if word in dictionary:
            index = dictionary[word]
            X[:, index] *= 2

    for word in negative_words:
        if word in dictionary:
            index = dictionary[word]
            X[:, index] *= 2


    clf = SVC(C=0.01, gamma=100, kernel='rbf')
    clf.fit(X, y)

    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    y_pred = clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/lilscolari_twitter.txt')

if __name__ == '__main__':
    main()
