"""
Author     : Cameron Scolari
Date       : 10/3/2024
Description: Polynomial Regression
This code was adapted from course material by Jenna Wiens (UMichigan).
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import time


######################################################################
# classes
######################################################################

class Data:

    def __init__(self, X=None, y=None):
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y


    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid:
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]


    def plot(self, **kwargs):
        """Plot data."""

        if 'color' not in kwargs:
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()


# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data


def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression():

    def __init__(self, m=1):
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
        """
        self.coef_ = None
        self.m_ = m


    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        # For part 7 (Comment out part 2 to use):
        # X_polynomial = X

        # for power in range(self.m_ + 1):
        #     X_polynomial = np.append(X_polynomial, X ** power, axis=1)

        # X_polynomial = X_polynomial[:, 1:]
        # Phi = X_polynomial

        # For part 2 (Comment out part 7 to use):
        ones = np.ones((n, 1))
        X = np.append(ones, X, axis=1)
        Phi = X

        return Phi


    def fit_SGD(self, X, y, alpha=None,
                eps=1e-10, tmax=1000000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares stochastic gradient descent.x

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            alpha   -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """

        if verbose:
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        alpha_input = alpha
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # SGD loop
        for t in range(tmax):
            if alpha_input is None:
                alpha = 1 / (t + 1)
            else:
                alpha = alpha_input

            # iterate through examples
            for i in range(n):
                phi = X[i,:]
                y_pred = np.dot(self.coef_, phi)

                self.coef_ = self.coef_ - alpha * (y_pred - y[i]) * phi

                y_pred = np.dot(self.coef_, phi)
                
                err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) < eps:
                break

            # debugging
            if verbose:
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print('number of iterations: %d' % (t+1))

        return self


    def fit(self, X, y):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        # X^T * X:
        X_t_X = X.T @ X

        # (X^T * X)^-1
        X_t_X_inv = np.linalg.pinv(X_t_X)

        # (X^T * X)^-1 * X^T * y
        self.coef_ = np.dot(X_t_X_inv @ X.T, y)

    def predict(self, X):
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception('Model not initialized. Perform a fit first.')

        X = self.generate_polynomial_features(X) # map features

        y = X @ self.coef_

        return y


    def cost(self, X, y):
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """

        n = len(y)

        predicted_y = self.predict(X)
        difference = predicted_y - y

        cost = (1/2) * np.sum((difference)**2)

        return cost


    def rms_error(self, X, y):
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """

        n = len(y)
        cost = self.cost(X, y)
        error = np.sqrt((2 * cost) / n)

        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main():
    # toy data
    X = np.array([2]).reshape((1,1))          # shape (n,d) = (1L,1L)
    y = np.array([3]).reshape((1,))           # shape (n,) = (1L,)
    coef = np.array([4,5]).reshape((2,))      # shape (d+1,) = (2L,), 1 extra for bias

    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')


    # part 1: plot train and test data
    print('Visualizing data...')
    train_data.plot()
    test_data.plot()


    # parts 2-6: main code for linear regression
    print('Investigating linear regression...')

    # model
    model = PolynomialRegression()

    # test part 2 -- soln: [[1 2]]
    print(model.generate_polynomial_features(X))

    # test part 3 -- soln: [14]
    model.coef_ = coef
    print(model.predict(X))

    # test part 4, bullet 1 -- soln: 60.5
    print(model.cost(X, y))

    # test part 4, bullets 2-3
    # for alpha = 0.01, soln: theta = [2.44; -2.82]
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    print('sgd solution: %s' % str(model.coef_))

    # test part 5 -- soln: theta = [2.45; -2.82]
    model.fit(train_data.X, train_data.y)
    print('closed_form solution: %s' % str(model.coef_))

    # non-test code (YOUR CODE HERE)

    # Checking runtime of SGD:
    time1 = time.time()
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    time2 = time.time()
    elapsed_time = time2 - time1
    print(f"{elapsed_time} seconds for SGD")

    # Checking runtime of closed-form:
    time1 = time.time()
    model.fit(train_data.X, train_data.y)
    time2 = time.time()
    elapsed_time = time2-time1
    print(f"{elapsed_time} seconds for closed-form")


    # 10^-1:
    model.fit_SGD(train_data.X, train_data.y, 1)
    print("model 1 coefficient:", model.coef_)

    # 10^-2:
    model.fit_SGD(train_data.X, train_data.y, 0.1)
    print("model 2 coefficient:", model.coef_)

    # 10^-3:
    model.fit_SGD(train_data.X, train_data.y, 0.01)
    print("model 3 coefficient:", model.coef_)

    # 10^-4:
    model.fit_SGD(train_data.X, train_data.y, 0.001)
    print("model 4 coefficient:", model.coef_)


    # Testing my learning rate:
    model.fit_SGD(train_data.X, train_data.y)
    print("coefficients with my learning rate:", model.coef_)


    # parts 7-9: main code for polynomial regression
    print('Investigating polynomial regression...')

    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([4,5,6]).reshape((3,))   # shape (3L,), 1 bias + 3 coeffs

    # test part 7 -- soln: [[1 2 4]]
    model = PolynomialRegression(m)
    print(model.generate_polynomial_features(X))

    # test part 8 -- soln: 35.0
    model.coef_ = coefm
    print(model.rms_error(X, y))

    # non-test code (YOUR CODE HERE)

    # part 9:
    train_rms_arr = np.array([])
    test_rms_arr = np.array([])
    m_arr = np.arange(11)

    for num in m_arr:
        m = num
        model = PolynomialRegression(m)
        model.fit(train_data.X, train_data.y)
        train_rms = model.rms_error(train_data.X, train_data.y)
        train_rms_arr = np.append(train_rms_arr, [train_rms])
        test_rms = model.rms_error(test_data.X, test_data.y)
        test_rms_arr = np.append(test_rms_arr, [test_rms])
    
    print(m_arr)
    print(train_rms_arr)
    print(test_rms_arr)

    plt.scatter(m_arr, train_rms_arr)
    plt.scatter(m_arr, test_rms_arr)
    labels = ["Training Data", "Test Data"]
    plt.legend(labels=labels)
    plt.xlabel('Polynomial Degree (m)', fontsize = 16)
    plt.ylabel('RMSE', fontsize = 16)
    plt.show()
    

    print('Done!')


if __name__ == "__main__":
    main()
