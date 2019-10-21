#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:49:09 2019

@author: oem
"""
import numpy as np


def add_to_list_in_dict(D, key, val):
        """
        Will add a value to a list within a dictionary if the key doesn't
        exist. If it does then this will simply append to that list.
        """
        if key in D:
            D[key].append(val)
        else:
            D[key] = [val]

def constructNN(pos, maxDist=0.3):
    mask = np.ones((len(pos), len(pos)), dtype=bool)
    NN = np.zeros((len(pos), len(pos)), dtype=float)
    for I in range(len(pos)):
        for J in range(I):
            NN[I, J] = np.linalg.norm(pos[I] - pos[J])
            NN[J, I] = NN[I, J]
            mask[I, J] = NN[I, J] < maxDist
            mask[J, I] = mask[I, J]
    return NN, mask


#data = [0.8, 2.2, 2.4, 0.9, 1.3, 1.2, 1, 2.3, 10]


def getSinglePointNeighbours(data, ind,  NN):
    clusteredPoints = [j for j in range(len(data)) if NN[ind, j]]
    return clusteredPoints


def clusterSingleGroup(data, ind, NN, pointsInCluster=[],
                       indsCompleted=[]):

    newClusteredPoints = getSinglePointNeighbours(data, ind, NN)

    for pnt in newClusteredPoints:
        if pnt not in pointsInCluster:  pointsInCluster.append(pnt)

    indsCompleted.append(ind)
    indsToComplete = [i for i in pointsInCluster if i not in indsCompleted]

    if len(indsToComplete) == 0:
        return pointsInCluster
    else:
        return clusterSingleGroup(data, indsToComplete[0], NN,
                                  pointsInCluster, indsCompleted)


def clusterAllPoints(data, NN, remainingPoints, ind=0, currClust=0,
                     clusters={}):

    pointsInCluster = clusterSingleGroup(data, ind, NN, [], [])
    clusters[currClust] = pointsInCluster

    for pnt in pointsInCluster:
        remainingPoints.remove(pnt)

    if len(remainingPoints) == 0:
        return clusters
    else:
        ind = remainingPoints[0]
        currClust += 1
        return clusterAllPoints(data, NN, remainingPoints, ind, currClust,
                                clusters)


def getClustID(clusters, ind):
    """
    Will find the ID of the cluster to which the index belongs
    """
    for clustI in clusters:
        if ind in clusters[clustI]:
            return clustI


def handle_bad_cluster(clusters, clustI, NN):
    """
    Will handle the points in a bad cluster.
    """
    badPoints = clusters[clustI]

    for indI, ind in enumerate(badPoints):
        # Get the nearest neighbour
        neighbours = NN[ind]
        inds = np.arange(len(neighbours))
        mask = neighbours > 0
        for i in badPoints:
            mask[i] = False

        inds = inds[mask]
        neighbours = neighbours[mask]
        nearestRep = inds[np.argmin(neighbours)]

        # Find the cluster of the nearest neighbour
        clustID = getClustID(clusters, nearestRep)

        # Add the point to the nearest neighbour cluster
        clusters[clustID].append(ind)


    clusters.pop(clustI)



def handle_outliers(clusters, NN, numPointsAllowed=5):
    """
    Will handle the outliers in the clusters (the clusters with only a few
    points in their cluster).
    """
    badClusters = [clustI for clustI in clusters
                   if len(clusters[clustI]) < numPointsAllowed]

    for badClustID in badClusters:
        handle_bad_cluster(clusters, badClustID, NN)

    if len(badClusters) == 0:
        return clusters
    else:
        return clusters


def getClusters(data, maxDist, handleOutliers=True):
    """
    Will use the above recursive functions to cluster the data points using a
    variant of the DBSCAN algorithm.
    """
    NN, mask = constructNN(data, maxDist)
    clusters = clusterAllPoints(data, mask, list(range(len(data))))

    if handleOutliers:
        clusters = handle_outliers(clusters, NN, 5)

    clustersData = {i: [data[j] for j in clusters[i]] for i in clusters}
    return clustersData, clusters


# Plot the data
def plotClusters(clusters, data):
    import matplotlib.pyplot as plt
    f, a = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'k', '#a6cee3',
              '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
              '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
              '#6a3d9a', '#ffff99']


    a.plot(data, np.ones(len(data)), 'ko')
    a.set_ylim([0.45, 1.55])

    print("\n\n")
    for clustI in clusters:
        print(np.std(clusters[clustI]))
        for x in clusters[clustI]:
            #plotCount += 1
            a.plot(x, 1,
                   'o', color=colors[clustI])