"""
Author     : Jack Seymour, Cameron Scolari, and Maria Dominguez
Date       : 12/10/2024
Description: Gaussian Process Regression
This code contains all the functions and methods needed to train, fit, and
visualize a Gaussian Process Regression model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import seaborn as sns

class Data:
    def __init__(self, X=None, y=None):
        """
        Data class.

        Attributes
        --------------------
            X -- numpy array of shape (n,d), features
            y -- numpy array of shape (n,), targets
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
        df = pd.read_csv(f)

        sns.pairplot(df, vars=["Engine Size(L)", "Fuel Consumption Comb (L/100 km)", "CO2 Emissions(g/km)"])
        plt.show()
            
        numeric_columns = ['Engine Size(L)', 'Fuel Consumption City (L/100 km)', 
                        'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
                        'Fuel Consumption Comb (mpg)', 'Cylinders']
            
        categorical_columns = ['Vehicle Class', 'Transmission', 'Fuel Type']
        encoder = OneHotEncoder(drop='first')
        encoded_data = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

        final_df = pd.concat([df, encoded_df], axis=1)

        X = final_df[numeric_columns + list(encoded_df.columns)]
        y = df['CO2 Emissions(g/km)']

        self.X = X.values
        self.y = y.values        

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
    """
    Wrapper function for load() method from Data class.

    Parameters
    --------------------
        filename -- string, filename
    """
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs):
    """
    Wrapper function for plot() method from Data class.

    Parameters
    --------------------
        X -- numpy array of shape (n,d), features
        y -- numpy array of shape (n,), targets
    """
    data = Data(X, y)
    data.plot(**kwargs)

def main():
    """
    Main function containing the code for sampling the data, training a model,
    fitting a model, running predictions with that model, visualizing results
    from that model, and calculating statistics from that model.
    """

    np.random.seed(42)

    data = load_data('co2.csv')
    
    # change to adjust sample size
    n=2000

    indices = np.random.choice(n, n, replace=False)

    X_train, X_test, y_train, y_test = train_test_split(data.X[indices], data.y[indices], test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Number of observations in training data: {len(X_train)} and in the testing data: {len(X_test)}")

    X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
    X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test

    gpr = GaussianProcessRegressor(random_state=42)

    param_distributions = {
        'kernel': [1.0 * RBF(length_scale=3.24) + 0.5 * DotProduct(sigma_0=50) + 0.1 * WhiteKernel(noise_level=10)],
        'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]
    }

    random_search = RandomizedSearchCV(
        estimator=gpr,
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        random_state=42,
        scoring='r2'
    )

    random_search.fit(X_train, y_train)

    gpr = random_search.best_estimator_

    print("Fitting the model...")
    gpr.fit(X_train, y_train)
    print("Model fitted.")
    print(gpr.kernel_)

    score = gpr.score(X_train, y_train)
    print(f"The R^2 value of the model on the training data was {score}")

    score = gpr.score(X_test, y_test)
    print(f"The R^2 value of the model on the testing data was {score}")

    y_mean, y_std = gpr.predict(X_test, return_std=True)

    y_upper = y_mean + 1.96 * y_std
    y_lower = y_mean - 1.96 * y_std


    y_training_mean = gpr.predict(X_train)
    train_error = mean_squared_error(y_train, y_training_mean)
    test_error = mean_squared_error(y_test, y_mean)

    print(f"Training mean squared error: {train_error} and test mean squared error: {test_error}")

    '''
    # Use this code to handle outliers. With current model, too many outliers are removed.

    outlier_threshold = 3
    mask = np.abs(y_test - y_mean) < outlier_threshold * y_std
    X_test_cleaned, y_test_cleaned = X_test[mask], y_test[mask]
    y_mean_cleaned, y_upper_cleaned, y_lower_cleaned = y_mean[mask], y_upper[mask], y_lower[mask]

    plt.fill_between(y_test_cleaned, y_lower_cleaned, y_upper_cleaned, color='orange', alpha=0.5, label='95% Confidence Interval')
    plt.scatter(y_test_cleaned, y_mean_cleaned, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test_cleaned), max(y_test_cleaned)], [min(y_test_cleaned), max(y_test_cleaned)], color='red', linestyle='--')
    '''

    plt.figure(figsize=(10, 6))

    # use CI to see how often the true value is within the predicted CI
    within_ci = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    total = len(y_test) 
    print(f'{(within_ci / total) * 100}% of actual values are within the 95% CI.')

    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_mean_sorted = y_mean[sorted_indices]
    y_lower_sorted = y_lower[sorted_indices]
    y_upper_sorted = y_upper[sorted_indices]

    plt.scatter(y_test, y_mean, color='blue', label='Predicted vs Actual')
    plt.fill_between(y_test_sorted, y_lower_sorted, y_upper_sorted, color='orange', alpha=0.5, label='95% Confidence Interval')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual CO2 Emissions (g/km)')
    plt.ylabel('Predicted CO2 Emissions (g/km)')
    plt.title('Predicted vs Actual CO2 Emissions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()