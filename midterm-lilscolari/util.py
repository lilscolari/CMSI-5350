"""
Author: Cameron Scolari
Date: 10/31/2024
Description: Utilities for Famous Faces
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import fetch_lfw_people


lfw_imageSize = (50,37)

######################################################################
# data utilities
######################################################################

# Fixes URLError when downloading sklearn LFW dataset.
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


def get_lfw_data():
    """
    Fetch LFW (Labeled Faces in the Wild) dataset.

    Warning: This will take a long time the first time you run it.
    It will download data onto disk but then will use the local copy thereafter.

    Returns
    --------------------
        X -- numpy array of shape (n,d), features (each row is one image)
        y -- numpy array of shape (n,), targets
             elements are integers in [0, num_classes-1]
    """

    global X, n, d, y, h, w

    lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.4)
    # TODO: if errors downloading dataset, download manually and add data_home hyperparameter to path with data from https://stackoverflow.com/questions/53433512/load-fetch-lfw-people-using-proxy
    n, h, w = lfw_people.images.shape
    X = lfw_people.data
    d = X.shape[1]
    y = lfw_people.target
    num_classes = lfw_people.target_names.shape[0]

    print('Total dataset size:')
    print('\tnum_samples: %d' % n)
    print('\tnum_features: %d' % d)
    print('\tnum_classes: %d' % num_classes)

    return X, y


def show_image(im, size=lfw_imageSize):
    """
    Open a new window and display the image.

    Parameters
    --------------------
        im   -- numpy array of shape (d,), image
        size -- tuple (i,j), i and j are positive integers such that i * j = d
                default to the right value for LFW dataset
    """

    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap=cm.gray)
    plt.show(block=True)


def plot_gallery(images, title='plot', subtitles=[],
                 h=50, w=37, n_row=3, n_col=4):
    """
    Plot array of images.

    Parameters
    --------------------
        images       -- numpy array of shape (12,d), images (one per row)
        title        -- title, title for entire plot
        subtitles    -- list of 12 strings or empty list, subtitles for subimages
        h, w         -- ints, image sizes
        n_row, n_col -- ints, number of rows and columns for plot
    """

    plt.figure(title, figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row*n_col)):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show(block=True)


def limit_pics(X, y, classes, nim):
    """
    Select subset of images from dataset.
    User can specify desired classes and desired number of images per class.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), features
        y       -- numpy array of shape (n,), targets
        classes -- list of ints, subset of target classes to retain
        nim     -- int, number of images desired per class

    Returns
    --------------------
        X1      -- numpy array of shape (nim * len(classes), d), subset of X
        y1      -- numpy array of shape (nim * len(classes),), subset of y
    """

    n, d = X.shape
    k = len(classes)
    X1 = np.zeros((k*nim, d), dtype=float)
    y1 = np.zeros(k*nim, dtype=int)
    index = 0
    for ni, i in enumerate(classes):      # for each class
        count = 0                           # count how many samples in class so far
        for j in range(n):                 # look over the data
            if count < nim and y[j] == i: # element of class
                X1[index] = X[j]
                y1[index] = ni
                index += 1
                count += 1
    return X1, y1


######################################################################
# sampling utilities
######################################################################

def random_sample_2d(mu, sigma):
    """
    Randomly sample point from a normal distribution.

    Parameters
    --------------------
        mu    -- numpy array of shape (2,), mean along each dimension
        sigma -- numpy array of shape (2,), standard deviation along each dimension

    Returns
    --------------------
        point -- numpy array of shape (2,), sampled point
    """

    x = np.random.normal(mu[0], sigma[0])
    y = np.random.normal(mu[1], sigma[1])
    return np.array([x,y])

######################################################################
# PCA utilities
######################################################################

def vec_to_image(x, size=lfw_imageSize):
    """
    Take an eigenvector and make it into an image.

    Parameters
    --------------------
        x    -- numpy array of shape (d,), eigenvector
        size -- tuple (i,j), i and j are positive integers such that i * j = d
                default to the right value for LFW dataset

    Returns
    --------------------
        im   -- numpy array of shape size, image
    """
    im = x/np.linalg.norm(x)
    im = im*(256./np.max(im))
    im.resize(*size)
    return im


def get_rep_image(X, y, label):
    """
    Get first image for each label.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features
        y     -- numpy array of shape (n,), targets
        label -- string, label

    Returns
    --------------------
        im    -- numpy array, image
    """
    tmp = X[y == label,:]
    return vec_to_image(tmp[0,:])


def PCA(X):
    """
    Perform Principal Component Analysis.
    This version uses SVD for better numerical performance when d >> n.

    Note that because the covariance matrix Sigma = XX^T is positive semi-definite,
    all its eigenvalues are non-negative.  So we can sort by l rather than |l|.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), features

    Returns
    --------------------
        U      -- numpy array of shape (d,d), d d-dimensional eigenvectors
                  each column is a unit eigenvector; columns are sorted by eigenvalue
        mu     -- numpy array of shape (d,), mean of input data X
    """
    n, d = X.shape
    mu = np.mean(X, axis=0)
    x, l, v = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    U = np.array([vi/1.0 \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return U, mu


def apply_PCA_from_Eig(X, U, l, mu):
    """
    Project features into lower-dimensional space.

    Parameters
    --------------------
        X  -- numpy array of shape (n,d), n d-dimensional features
        U  -- numpy array of shape (d,d), d d-dimensional eigenvectors
              each column is a unit eigenvector; columns are sorted by eigenvalue
        l  -- int, number of principal components to retain
        mu -- numpy array of shape (d,), mean of input data X

    Returns
    --------------------
        Z   -- numpy matrix of shape (n,l), n l-dimensional features
               each row is a sample, each column is one dimension of the sample
        Ul  -- numpy matrix of shape (d,l), l d-dimensional eigenvectors
               each column is a unit eigenvector; columns are sorted by eigenvalue
               (Ul is a subset of U, specifically the d-dimensional eigenvectors
                of U corresponding to largest l eigenvalues)
    """
    ### ========== TODO: START ========== ###
    # TODO: in one line, subtract the mean from X and multiply by Ul
    Ul = np.asmatrix(U[:,:l])

    Z = (X - mu) @ Ul
    ### ========== TODO: END ========== ###
    return Z, Ul


def reconstruct_from_PCA(Z, U, mu):
    """
    Reconstruct features from eigenvectors.

    Parameters
    --------------------
        Z     -- numpy matrix of shape (n,l), n l-dimensional features
        U     -- numpy matrix of shape (d,l), l d-dimensional eigenvectors
                 each column is a unit eigenvector; columns are sorted by eigenvalue
        mu    -- numpy array of shape (d,), mean of input data X

    Returns
    --------------------
        X_rec -- numpy matrix of shape (n,d), reconstructed features
    """
    ### ========== TODO: START ========== ###
    # TODO: in one line, multiply by U transpose and add the mean back in
    X_rec = (Z @ U.T) + mu
    ### ========== TODO: END ========== ###
    return X_rec
