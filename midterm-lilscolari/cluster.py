"""
Author: Cameron Scolari
Date: 10/31/2024
Description: This is the file that should be edited for the extra credit.
"""

import collections
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


######################################################################
# classes
######################################################################

class Point(object):
    
    def __init__(self, name, label, attrs):
        """
        A data point.
        
        Attributes
        --------------------
            name  -- string, name
            label -- string, label
            attrs -- numpy array, features
        """
        
        self.name = name
        self.label = label
        self.attrs = attrs
    
    
    #============================================================
    # utilities
    #============================================================
    
    def distance(self, other):
        """
        Return Euclidean distance of this point with other point.
        
        Parameters
        --------------------
            other -- Point, point to which we are measuring distance
        
        Returns
        --------------------
            dist  -- float, Euclidean distance
        """
        # Euclidean distance metric
        return np.linalg.norm(self.attrs - other.attrs)
    
    
    def __str__(self):
        """
        Return string representation.
        """
        return '%s: (%s, %s)' % (self.name, str(self.attrs), self.label)


class Cluster(object):
    
    def __init__(self, points):
        """
        A cluster (set of points).
        
        Attributes
        --------------------
            points -- list of Points, cluster elements
        """        
        self.points = points
    
    
    def __str__(self):
        """
        Return string representation.
        """
        s = ''
        for point in self.points:
            s += str(point)
        return s
        
    #============================================================
    # utilities
    #============================================================
    
    def purity(self):
        """
        Compute cluster purity.
        
        Returns
        --------------------
            n           -- int, number of points in this cluster
            num_correct -- int, number of points in this cluster
                                with label equal to most common label in cluster
        """        
        labels = []
        for p in self.points:
            labels.append(p.label)
        
        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)
    
    
    def centroid(self):
        """
        Compute centroid of this cluster.
        
        Returns
        --------------------
            centroid -- Point, centroid of cluster
        """
        
        ### ========== TODO: START ========== ###
        # set the centroid label to any value (e.g. the most common label in this cluster)

        labels = [point.label for point in self.points]

        most_common_label = stats.mode(labels)[0]

        all_attributes = np.array([point.attrs for point in self.points])
        centroid_attributes = np.mean(all_attributes, axis=0)

        centroid = Point("centroid", most_common_label, centroid_attributes)

        return centroid
        ### ========== TODO: END ========== ###
    
    
    def medoid(self):
        """
        Compute medoid of this cluster, that is, the point in this cluster
        that is closest to all other points in this cluster.
        
        Returns
        --------------------
            medoid -- Point, medoid of this cluster
        """
        
        smallest_total_distance = np.inf
        medoid = None

        for point in self.points:
            total_distances = sum(point.distance(other) for other in self.points if other != point)

            if total_distances < smallest_total_distance:
                medoid = point
                smallest_total_distance = total_distances

        ### ========== TODO: START ========== ###
        return medoid
        ### ========== TODO: END ========== ###
    
    
    def equivalent(self, other):
        """
        Determine whether this cluster is equivalent to other cluster.
        Two clusters are equivalent if they contain the same set of points
        (not the same actual Point objects but the same geometric locations).
        
        Parameters
        --------------------
            other -- Cluster, cluster to which we are comparing this cluster
        
        Returns
        --------------------
            flag  -- bool, True if both clusters are equivalent or False otherwise
        """
        
        if len(self.points) != len(other.points):
            return False
        
        matched = []
        for point1 in self.points:
            for point2 in other.points:
                if point1.distance(point2) == 0 and point2 not in matched:
                    matched.append(point2)
        return len(matched) == len(self.points)


class ClusterSet(object):

    def __init__(self):
        """
        A cluster set (set of clusters).
        
        Parameters
        --------------------
            members -- list of Clusters, clusters that make up this set
        """
        self.members = []
    
    
    #============================================================
    # utilities
    #============================================================    
    
    def centroids(self):
        """
        Return centroids of each cluster in this cluster set.
        
        Returns
        --------------------
            centroids -- list of Points, centroids of each cluster in this cluster set
        """
        
        ### ========== TODO: START ========== ###
        centroids = []
        for member in self.members:
            centroids.append(member.centroid())
        return centroids
        ### ========== TODO: END ========== ###
    
    
    def medoids(self):
        """
        Return medoids of each cluster in this cluster set.
        
        Returns
        --------------------
            medoids -- list of Points, medoids of each cluster in this cluster set
        """
        
        ### ========== TODO: START ========== ###
        medoids = []
        for member in self.members:
            medoids.append(member.medoid())
        return medoids
        ### ========== TODO: END ========== ###
    
    
    def score(self):
        """
        Compute average purity across clusters in this cluster set.
        
        Returns
        --------------------
            score -- float, average purity
        """
        
        total_correct = 0
        total = 0
        for c in self.members:
            n, n_correct = c.purity()
            total += n
            total_correct += n_correct
        return total_correct / float(total)
    
    
    def equivalent(self, other):
        """ 
        Determine whether this cluster set is equivalent to other cluster set.
        Two cluster sets are equivalent if they contain the same set of clusters
        (as computed by Cluster.equivalent(...)).
        
        Parameters
        --------------------
            other -- ClusterSet, cluster set to which we are comparing this cluster set
        
        Returns
        --------------------
            flag  -- bool, True if both cluster sets are equivalent or False otherwise
        """
        
        if len(self.members) != len(other.members):
            return False
        
        matched = []
        for cluster1 in self.members:
            for cluster2 in other.members:
                if cluster1.equivalent(cluster2) and cluster2 not in matched:
                    matched.append(cluster2)
        return len(matched) == len(self.members)
    
    
    #============================================================
    # manipulation
    #============================================================

    def add(self, cluster):
        """
        Add cluster to this cluster set (only if it does not already exist).
        
        If the cluster is already in this cluster set, raise a ValueError.
        
        Parameters
        --------------------
            cluster -- Cluster, cluster to add
        """
        
        if cluster in self.members:
            raise ValueError
        
        self.members.append(cluster)


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k):
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO:x START ========== ###
    # hint: use np.random.choice
    initial_points = np.random.choice(points, k, replace = False)
    return initial_points
    ### ========== TODO: END ========== ###


def cheat_init(points):
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO: START ========== ###


    label_groups = dict()
    for point in points:
        try:
            label_groups[point.label].append(point)
        except:
            label_groups[point.label] = []
            label_groups[point.label].append(point)

    initial_points = []

    for points in label_groups.values():
        new_cluster = Cluster(points)
        medoid = new_cluster.medoid()
        initial_points.append(medoid)

    return initial_points
    ### ========== TODO: END ========== ###


def kMeans(points, k, init='random', plot=False):
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO: START ========== ###
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to 
    #       create new Cluster objects and a new ClusterSet object. Then
    #       update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    if init == 'cheat':
        initial_points = cheat_init(points)
    else:
        initial_points = random_init(points, k)
    clusters = [Cluster([point]) for point in initial_points]

    k_clusters = ClusterSet()
    k_clusters.members = clusters

    previous_assignment = None

    while True:
        current_assignment = []
        new_clusters = [[] for _ in range(k)]

        for point in points:
            distances = [point.distance(cluster.centroid()) for cluster in k_clusters.members]
            nearest_cluster_index = np.argmin(distances)
            new_clusters[nearest_cluster_index].append(point)
            current_assignment.append(nearest_cluster_index)

        updated_clusters = [Cluster(cluster) for cluster in new_clusters]

        new_cluster_set = ClusterSet()
        new_cluster_set.members = updated_clusters

        if current_assignment == previous_assignment:
            break

        k_clusters = new_cluster_set
        previous_assignment = current_assignment

        if plot:
            plot_clusters(k_clusters, "title", ClusterSet.centroids)
    return k_clusters
    ### ========== TODO: END ========== ###


def kMedoids(points, k, init='random', plot=False):
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    if init == 'cheat':
        initial_points = cheat_init(points)
    else:
        initial_points = random_init(points, k)
    clusters = [Cluster([point]) for point in initial_points]

    ### ========== TODO: START ========== ###
    k_clusters = ClusterSet()

    k_clusters.members = clusters

    previous_assignment = None

    while True:
        current_assignment = []

        new_clusters = [[] for _ in range(k)]

        for point in points:
            distances = [point.distance(cluster.medoid()) for cluster in k_clusters.members]

            nearest_cluster_index = np.argmin(distances)

            new_clusters[nearest_cluster_index].append(point)
            current_assignment.append(nearest_cluster_index)


        updated_clusters = []
        for cluster_points in new_clusters:
            cluster = Cluster(cluster_points)
            medoid = cluster.medoid()
            updated_clusters.append(cluster)


        new_cluster_set = ClusterSet()
        new_cluster_set.members = updated_clusters

        if current_assignment == previous_assignment:
            break

        k_clusters = new_cluster_set
        previous_assignment = current_assignment

        if plot:
            plot_clusters(k_clusters, "title", ClusterSet.medoids)
    return k_clusters
    ### ========== TODO: END ========== ###


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()
