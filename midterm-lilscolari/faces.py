"""
Author: Cameron Scolari
Date: 10/31/2024
Description: This file has a bunch of methods and code to run PCA and clustering techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import util
# TODO: change cluster_5350 to cluster if you do the extra credit
from cluster_5350.cluster import *
# EC 3/4 WORKS: from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y):
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n):
        images[y[i]].append(X[i,:])

    points = []
    for face in images:
        count = 0
        for im in images[face]:
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def generate_points_2d(N, seed=1234):
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma):
        label += 1
        for i in range(N):
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# main
######################################################################

def main():
    X, y = util.get_lfw_data()

    for image in X[:10]:
        util.show_image(image)
    
    # Mean of all the images:
    average_image = np.zeros(len(X[0]))
    for image in X:
        for index, number in enumerate(image):
            average_image[index] += number

    average_image /= len(X[0])
    util.show_image(average_image)

    U, mu = util.PCA(X)
    util.plot_gallery([util.vec_to_image(U[:,i]) for i in range(12)])

    l_list = [1, 10, 50, 100, 500, 1288]

    for l in l_list:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        reconstructed_X = util.reconstruct_from_PCA(Z, Ul, mu)
        util.plot_gallery([util.vec_to_image(reconstructed_X[row]) for row in range(12)])

    #===============================================
    # (Optional) part 2: test Cluster implementation
    # centroid: [ 1.04022358  0.62914619]
    # medoid:   [ 1.05674064  0.71183522]

    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    cluster = Cluster(sim_points)
    print('centroid:', cluster.centroid().attrs)
    print('medoid:', cluster.medoid().attrs)

    # test kMeans and kMedoids implementation using toy dataset
    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    k = 3

    # cluster using random initialization
    kmeans_clusters = kMeans(sim_points, k, init='random', plot=True)
    kmedoids_clusters = kMedoids(sim_points, k, init='random', plot=True)

    # cluster using cheat initialization
    kmeans_clusters = kMeans(sim_points, k, init='cheat', plot=True)
    kmedoids_clusters = kMedoids(sim_points, k, init='cheat', plot=True)


    np.random.seed(1234)

    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    k_means_all_trials = []
    k_medoids_all_trials = []
    k_means_run_times = []
    k_medoids_run_times = []

    for _ in range(10):
        # Use k=4 because 4 selected classes:
        start_time = time.time()
        k_clusters = kMeans(points, 4)
        end_time = time.time()
        k_means_run_times.append(end_time - start_time)
        k_means_all_trials.append(k_clusters.score())
        start_time = time.time()
        k_clusters = kMedoids(points, 4)
        end_time = time.time()
        k_medoids_run_times.append(end_time - start_time)
        k_medoids_all_trials.append(k_clusters.score())

    average_k_means_runtime = sum(k_means_run_times) / len(k_means_run_times)
    average_k_medoids_runtime = sum(k_medoids_run_times) / len(k_medoids_run_times)

    print(f"Average K-Means Run Time: {average_k_means_runtime}")
    print(f"Average K-Medoids Run Time: {average_k_medoids_runtime}")

    print(f"K-Means: average: {np.average(k_means_all_trials)}, min: {min(k_means_all_trials)}, max: {max(k_means_all_trials)}")
    print(f"K-Medoids: average: {np.average(k_medoids_all_trials)}, min: {min(k_medoids_all_trials)}, max: {max(k_medoids_all_trials)}")

    X1, y1 = util.limit_pics(X, y, [4, 13], 40)
    # apply PCA on entire image dataset so use X:
    u, mu = util.PCA(X)

    l_list = range(1, 42, 2)

    k_means_scores = []
    k_medoids_scores = []

    for l in l_list:
        Z, Ul = util.apply_PCA_from_Eig(X1, u, l, mu)
        # Use Z because we want to compute scores of lower dimension:
        points = build_face_image_points(Z, y1)

        k_clusters = kMeans(points, 2, init='cheat')
        k_means_scores.append(k_clusters.score())
        k_clusters = kMedoids(points, 2, init='cheat')
        k_medoids_scores.append(k_clusters.score())

    plt.plot(l_list, k_means_scores, label='k-means')
    plt.plot(l_list, k_medoids_scores, label='k-medoids')
    plt.xlabel('Number of Components (l)')
    plt.ylabel('Clustering Score')
    plt.title('Clustering Score vs. Number of Components')
    plt.legend()
    plt.show()

    high_score = -np.inf
    high_pair = None
    low_score = np.inf
    low_pair = None

    u, mu = util.PCA(X)

    unique_labels = np.unique(y)

    for index, label1 in enumerate(unique_labels):
        for label2 in unique_labels[index + 1:]:
            X1, y1 = util.limit_pics(X, y, [label1, label2], 40)
            Z, Ul = util.apply_PCA_from_Eig(X1, u, 50, mu)
            points = build_face_image_points(Z, y1)
            k_clusters = kMeans(points, 2, init='cheat')
            score = k_clusters.score()
            if score > high_score:
                high_score = score
                high_pair = (label1, label2)
            if score < low_score:
                low_score = score
                low_pair = (label1, label2)

    print(f"Low Score (Least discriminative): {low_score}")
    util.show_image(X[low_pair[0]])
    util.show_image(X[low_pair[1]])
    print(f"High Score (Most discriminative): {high_score}")
    util.show_image(X[high_pair[0]])
    util.show_image(X[high_pair[1]])

if __name__ == "__main__":
    main()
