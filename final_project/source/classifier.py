"""
Author     : Jack Seymour, Cameron Scolari, and Maria Dominguez
Date       : 12/10/2024
Description: Gaussian Process Classification
This code contains all the functions and methods needed to train, fit, and
visualize a Gaussian Process Classification model.
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

class DataLoader:
    def __init__(self, filename):
        """
        Data Loader class.

        Attributes
        --------------------
            filename -- string, filename
        """
        self.filename = filename

    def load_data(self):
        """
        Load csv file into X array of features and y array of labels.
        """
        dir = os.path.dirname(__file__)
        filepath = os.path.join(dir, '..', 'data', self.filename)
        df = pd.read_csv(filepath)

        feature_columns = df.columns[1:-1]  # All columns except the first and the last
        target_column = df.columns[-1]  # The last column

        X = df[feature_columns].values
        y = df[target_column].values
        return X, y

class DataPlotter:
    """
    A utility class for visualizing data points in a scatter plot.

    Methods:
    --------
    plot(X, y, **kwargs):
        Creates a scatter plot of the data points, coloring them according to the target values.
    """
    @staticmethod
    def plot(X, y, **kwargs):
        """
        Creates a scatter plot of the given data.

    Parameters
    --------------------
        X -- numpy array of shape (n,d), features
        y -- numpy array of shape (n,), targets
        **kwargs : dict, optional
            Additional keyword arguments to pass to `plt.scatter`.
            For example, 'color' to set a default color.
        """
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        plt.scatter(X[:, 0], X[:, 1], c=y, **kwargs)
        plt.xlabel('Feature 1', fontsize=16)
        plt.ylabel('Feature 2', fontsize=16)
        plt.show()

def preprocess_data(X):
    """
    Function for preprocessing the data.

    Parameters
    --------------------
        X -- numpy array of shape (n,d), features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_model(X_train, y_train):
    """
    Function for training the model.

    Parameters
    --------------------
        X -- numpy array of shape (n,d), features
        y -- numpy array of shape (n,), targets
    """
    gpc = GaussianProcessClassifier(random_state=134)
    param_distributions = {
        'kernel': [1.0 * RBF(), 1.0 * Matern()],
    }
    random_search = RandomizedSearchCV(
        estimator=gpc,
        param_distributions=param_distributions,
        n_iter=2,
        cv=2,
        random_state=134,
        scoring='accuracy'
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    return best_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Function for evaluating the model by printing performance statistics.

    Parameters
    --------------------
        model -- 
        X_train -- numpy array of shape (n of train data,d), features
        y_train -- numpy array of shape (n of train data,), targets
        X_test -- numpy array of shape (n of test data,d), features
        y_test -- numpy array of shape (n of test data), targets
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class

    print("Sample of y_test_prob:", y_test_prob[:10])
    print("Min and Max of y_test_prob:", y_test_prob.min(), y_test_prob.max())

    plt.hist(y_test_prob[y_test == 0], bins=20, alpha=0.5, label='No Fraud', color='blue')
    plt.hist(y_test_prob[y_test == 1], bins=20, alpha=0.5, label='Fraud', color='orange')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.show()

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Training accuracy: {train_accuracy}")
    print(f"Testing accuracy: {test_accuracy}")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show(block = True)

def main():
    """
    Main function containing the code for sampling the data, training a model,
    fitting a model, running predictions with that model, visualizing results
    from that model, and calculating statistics from that model.
    """

    np.random.seed(134)
    data_loader = DataLoader('creditcard.csv')
    X, y = data_loader.load_data()
    
    # Combine X and y into a single DataFrame for sampling
    df = pd.DataFrame(X)
    df['target'] = y

    # Plot visualization of data:
    value_counts = df['target'].value_counts()
    plt.figure(figsize=(8, 5))
    value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Target Column Distribution', fontsize=16)
    plt.xlabel('Target Values', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['No Fraud (0)', 'Fraud (1)'], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Sample 2000 rows
    df_sampled = df.sample(n=2000, random_state=134)

    # Print value counts:
    value_counts = df_sampled['target'].value_counts()
    print(value_counts)
    
    # Separate the features and target again
    X_sampled = df_sampled.drop(columns=['target']).values
    y_sampled = df_sampled['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=134)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()