"""
Author: Cameron Scolari
Date: 10/10/24
Description: Perceptron
"""

# This code was adapted course material by Tommi Jaakola (MIT).

# utilities
from util import *

# scikit-learn libraries
from sklearn.svm import SVC

######################################################################
# functions
######################################################################

def load_simple_dataset(start=0, outlier=False):
    """Simple dataset of three points."""

    #  dataset
    #     i    x^{(i)}      y^{(i)}
    #     1    (-1, 1)^T    1
    #     2    (0, -1)^T    -1
    #     3    (1.5, 1)^T   1
    #   if outlier is set, x^{(3)} = (12, 1)^T

    # data set
    data = Data()
    data.X = np.array([[ -1, 1],
                       [  0,-1],
                       [1.5, 1]])
    if outlier:
        data.X[2,:] = [12, 1]
    data.y = np.array([1, -1, 1])

    # circularly shift the data points
    data.X = np.roll(data.X, -start, axis=0)
    data.y = np.roll(data.y, -start)

    return data


def plot_perceptron(data, clf, plot_data=True, axes_equal=False, **kwargs):
    """Plot decision boundary and data."""
    assert isinstance(clf, Perceptron)

    # plot options
    if 'linewidths' not in kwargs:
        kwargs['linewidths'] = 2
    if 'colors' not in kwargs:
        kwargs['colors'] = 'k'

    # plot data
    if plot_data: data.plot()

    # axes limits and properties
    xmin, xmax = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
    ymin, ymax = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
    if axes_equal:
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # create a mesh to plot in
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))

    # determine decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # plot decision boundary
    Z = Z.reshape(xx.shape)
    CS = plt.contour(xx, yy, Z, [0], **kwargs)

    # legend
    if 'label' in kwargs:
        #plt.clabel(CS, inline=1, fontsize=10)
        CS.collections[0].set_label(kwargs['label'])

    plt.show()


######################################################################
# classes
######################################################################

class Perceptron:

    def __init__(self):
        """
        Perceptron classifier that keeps track of mistakes made on each data point.

        Attributes
        --------------------
            coef_     -- numpy array of shape (d,), feature weights
            mistakes_ -- numpy array of shape (n,), mistakes per data point
        """
        self.coef_ = None
        self.mistakes_ = None

    def fit(self, X, y, coef_init=None, verbose=False):
        """
        Fit the perceptron using the input data.

        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            y         -- numpy array of shape (n,), targets
            coef_init -- numpy array of shape (n,d), initial feature weights
            verbose   -- boolean, for debugging purposes

        Returns
        --------------------
            self      -- an instance of self
        """
        # get dimensions of data
        n,d = X.shape

        # initialize weight vector to all zeros
        if coef_init is None:
            self.coef_ = np.zeros(d)
        else:
            self.coef_ = coef_init

        # record number of mistakes we make on each data point
        self.mistakes_ = np.zeros(n)

        # debugging
        if verbose:
            print(f'\ttheta^0 = {self.coef_}')

        mistakes = True

        while mistakes:
            mistakes = False
            for i in range(n):
                if y[i] * np.dot(self.coef_.T, X[i]) <= 0:
                    self.coef_ = self.coef_ + y[i] * X[i]
                    self.mistakes_[i] += 1
                    if verbose:
                        print(f'Updated weights after mistake on sample {i}: {self.coef_}')
                    mistakes = True
        return self

    def predict(self, X):
        """
        Predict labels using perceptron.

        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features

        Returns
        --------------------
            y_pred    -- numpy array of shape (n,), predictions
        """
        return np.sign(np.dot(X, self.coef_))


######################################################################
# main
######################################################################

def main():

    #========================================
    # test simple data set

    # starting with data point x^(1) without outlier
    #   coef = [ 0.  1.], mistakes = [1. 0. 0.]
    # starting with data point x^(2) without outlier
    #   coef = [ 0.5  2. ], mistakes = [1. 1. 0.]
    # starting with data point x^(1) with outlier
    #   coef = [ 0.  1.], mistakes = [1. 0. 0.]
    # starting with data point x^(2) with outlier
    #   coef = [ 6.  7.], mistakes = [1. 6. 0.]
    clf = Perceptron()
    for outlier in (False, True):
        for start in (1, 2):
            text = 'starting with data point x^(%d) %s outlier' % \
                (start, 'with' if outlier else 'without')
            print(text)
            data = load_simple_dataset(start, outlier)
            clf.fit(data.X, data.y, verbose=False)
            print(f'\tcoef = {clf.coef_}, mistakes = {clf.mistakes_}')

    # training perceptron classifier with two initializations:
    train_data = load_data('perceptron_data.csv')
    clf.fit(train_data.X, train_data.y, coef_init=np.array([0, 0]))
    print(f'\tcoef = {clf.coef_}, mistakes = {clf.mistakes_}')
    clf.fit(train_data.X, train_data.y, coef_init=np.array([1, 0]))
    print(f'\tcoef = {clf.coef_}, mistakes = {clf.mistakes_}')


    # part 2: train a perceptron with two initializations (hint: use coef_init)
    #  coef = [3.202321 2.128344]
    #  coef = [0.911808 0.79204]


if __name__ == '__main__':
    main()
