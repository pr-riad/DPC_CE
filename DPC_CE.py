# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:56:51 2024

@author: riad9

This is a replication of Matlab code of DPC-CE 
(Density Peak Clustering with Connectivity Estimation，Knowledge-Based Systems, 2022)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score



def perform_assignation(rho, delta, ordrho, nneigh, dist, dc, NCLUST, icl):
    
    """
    # Example usage
    cl, halo = perform_assignation(rho, delta, ordrho, nneigh, dist, dc, NCLUST, icl)
    """
    ND = len(rho)
    cl = np.full(ND, -1)
    halo = np.full(ND, -1)

    print('Performing assignation')

    # Clustering non-center points, traverse by rho
    for i in range(ND):
        if cl[ordrho[i]] == -1:
            cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

    # Deal with halo
    for i in range(ND):
        halo[i] = cl[i]

    if NCLUST > 1:
        # Initialize bord_rho for every cluster = 0
        bord_rho = np.zeros(NCLUST)

        # Calculate average bord_rho for each cluster
        for i in range(ND - 1):
            for j in range(i + 1, ND):
                # Two close points but in different clusters
                if cl[i] != cl[j] and dist[i, j] <= dc:
                    rho_aver = (rho[i] + rho[j]) / 2.0  # The average rho is the threshold

                    if rho_aver > bord_rho[cl[i]]:
                        bord_rho[cl[i]] = rho_aver

                    if rho_aver > bord_rho[cl[j]]:
                        bord_rho[cl[j]] = rho_aver

    # For every cluster
    for i in range(NCLUST):
        nc = 0  # The number of all points in cluster
        nh = 0  # The number of core points in cluster
        for j in range(ND):
            if cl[j] == i:
                nc += 1
            if halo[j] == i:  # Non-outlier
                nh += 1

        print(f'CLUSTER: {i + 1} CENTER: {icl[i] + 1} ELEMENTS: {nc} CORE: {nh} HALO: {nc - nh}')

    return cl, halo



def find_cluster_centers(rho, delta, rhomin, deltamin):
    """
    # Example usage

    cl, icl, NCLUST = find_cluster_centers(rho, delta, rhomin, deltamin)

    Parameters
    ----------
    rh
    delta 
    rhomin 
    deltamin 

    Returns
    -------
    cl 
    icl 
    NCLUST 

    """
    
    ND = len(rho)
    cl = np.full(ND, -1)
    icl = []
    NCLUST = 0

    # Find cluster centers
    for i in range(ND):
        if rho[i] > rhomin and delta[i] > deltamin:
            NCLUST += 1
            cl[i] = NCLUST
            icl.append(i)

    print(f'NUMBER OF CLUSTERS: {NCLUST}')

    # Plot the decision graph
    plt.figure()
    plt.plot(rho, delta, 'o', markersize=5, markerfacecolor='k', markeredgecolor='k')
    plt.title('Decision Graph of DPC-CE', fontsize=12)
    plt.xlabel(r'$\rho$', fontsize=14)
    plt.ylabel(r'$\delta$', fontsize=14)
    plt.box(False)
    
    # Get the limits of the current figure
    a = plt.axis()
    xmin, xmax, ymin, ymax = a

    # Make grid
    options = {'gridx': 50, 'gridy': 50}
    X, Y = np.meshgrid(np.linspace(xmin, xmax, options['gridx']),
                       np.linspace(ymin, ymax, options['gridy']))

    # Make testing patterns covering the whole grid
    tst_data = np.vstack([X.ravel(), Y.ravel()])
    dec_fun = tst_data[0, :] * tst_data[1, :]
    Z = dec_fun.reshape(X.shape)

    # Smooth shading
    plt.contour(X, Y, Z, 1, colors='k')

    # Draw cluster centers with different colors
    cmap = plt.get_cmap('viridis')
    for i in range(NCLUST):
        ic = int((i * 64) / (NCLUST * 1))
        plt.plot(rho[icl[i]], delta[icl[i]], 'o', markersize=8, 
                 markerfacecolor=cmap(ic / 64), markeredgecolor=cmap(ic / 64))
        plt.contour(X, Y, Z, levels=[rho[icl[i]] * delta[icl[i]]], colors=[cmap(ic / 64)])

    plt.show()

    return cl, icl, NCLUST



def decision_graph(rho, delta):
    
    """
    # Example usage
    rho = np.random.rand(100) * 10  # Example data
    delta = np.random.rand(100) * 10  # Example data
    rhomin, deltamin, NCLUST = decision_graph(rho, delta)
    print(f'rhomin: {rhomin}, deltamin: {deltamin}, NCLUST: {NCLUST}')
    """
    
    # Generate the decision graph file
    with open('DECISION_GRAPH', 'w') as fid:
        for i in range(len(rho)):
            fid.write(f'{rho[i]:6.2f} {delta[i]:6.2f}\n')
    
    print('Generated file: DECISION_GRAPH')
    print('column 1: Density')
    print('column 2: Delta')
    
    # Plot the decision graph
    plt.figure()
    plt.plot(rho, delta, 'o', markersize=5, markerfacecolor='k', markeredgecolor='k')
    plt.title('Decision Graph of DPC-CE', fontsize=12)
    plt.xlabel(r'$\rho$', fontsize=14)
    plt.ylabel(r'$\delta$', fontsize=14)
    plt.box(False)
    plt.show()
    
    # Select a rectangle enclosing cluster centers
    print('Select a rectangle enclosing cluster centers')
    rect = plt.ginput(2)  # Get two points from the user
    rhomin = min(rect[0][0], rect[1][0])
    deltamin = min(rect[0][1], rect[1][1])
    
    # Initialize number of clusters
    NCLUST = 0
    
    return rhomin, deltamin, NCLUST


    

def centralize_and_scale(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled
"""
def centralize_and_scale(data):
    # Centralize the data by subtracting the mean
    data = data - np.mean(data, axis=0)
    # Scale the data by dividing by the maximum absolute value
    data = data / np.max(np.abs(data))
    return data
"""

def judge_nei(pair, dist, count, dc, dist_ther):
    turn = 0
    find_time = 0
    nei_all = [pair[0]]
    nei_new = nei_all
    nei_dist = 0

    while turn == 0:
        find_time += 1
        nei_temp = []

        if dist[pair[0], pair[1]] < dc * 2:
            turn = 1
            count[0] += 1

        for k in range(len(nei_new)):
            nei_add = np.where(dist[nei_new[k], :] < dist_ther)[0]
            nei_dist_max = np.mean(dist[nei_new[k], nei_add])
            if nei_dist_max > nei_dist:
                nei_dist = nei_dist_max
            nei_temp.extend(nei_add)

        nei_new = list(set(nei_temp) - set(nei_all))

        if pair[1] in nei_all:
            turn = 2
            count[1] += 1
        elif len(nei_new) < 1:
            turn = 3
            count[2] += 1

        nei_all.extend(nei_new)

    return turn, nei_all, find_time, count, nei_dist

################################ The main algo ################################
    
# Load the data from the text file
fourlines = np.loadtxt('twomoons.txt')

# Extract the data and labels
data = fourlines[:, :-1]
true_label = fourlines[:, -1]

    
# Get the dimensions of the data
dim = data.shape[1]
num = data.shape[0]
    
data = centralize_and_scale(data)

# Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=true_label, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=true_label, alpha=1, s=20)

# Compute the distance matrix
mdist = distance_matrix(data, data)

xx = mdist

# Determine the number of points
ND = int(max(xx[:, 1]))
NL = int(max(xx[:, 0]))
if NL > ND:
    ND = NL

# Number of pair distances
N = xx.shape[0]

# Initialize the distance matrix
dist = np.zeros((ND, ND))

# Fill the distance matrix
for i in range(N):
    ii = int(xx[i, 0]) - 1  # Adjusting for 0-based indexing
    jj = int(xx[i, 1]) - 1  # Adjusting for 0-based indexing
    dist[ii, jj] = xx[i, 2]
    dist[jj, ii] = xx[i, 2]

# Compute dc: 2% distance
percent = 2.0
print(f'average percentage of neighbours (hard coded): {percent:.6f}')

position = round(N * percent / 100)
sda = np.sort(xx[:, 2])
dc = sda[position]

print(f'Computing Rho with gaussian kernel of radius: {dc:.6f}')


# Initialize rho: density of points
rho = np.zeros(ND)

# Gaussian kernel within dc and cut_off
for i in range(ND - 1):
    for j in range(i + 1, ND):
        if dist[i, j] < dc:
            rho[i] += np.exp(-(dist[i, j] / dc) ** 2)
            rho[j] += np.exp(-(dist[i, j] / dc) ** 2)

# Find the max distance
maxd = np.max(dist)

# Rank rho by descending order
rho_sorted = np.sort(rho)[::-1]
ordrho = np.argsort(rho)[::-1]
rho_or = rho_sorted
ordrho_or = ordrho

# Deal with point with max rho
delta = np.full(ND, -1.0)
nneigh = np.zeros(ND, dtype=int)

# Compute the delta (relative distance), find nneigh for points
for ii in range(1, ND):
    delta[ordrho[ii]] = maxd
    for jj in range(ii):
        if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
            delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
            nneigh[ordrho[ii]] = ordrho[jj]

# Give max rho point max delta
delta[ordrho[0]] = np.max(delta)

# CES (distance punishment)
count = [0, 0, 0]  # record punishment
choose_num = 20
dist_ther1_ratio = 0.25
punish_ratio = 0.3

delta_sorted = np.sort(delta)[::-1]
orddelta = np.argsort(delta)[::-1]
choose_point = orddelta[:choose_num]
rho_choose_point = rho[choose_point]
ordrho_choose = np.argsort(rho_choose_point)[::-1]
pair_all = choose_point[ordrho_choose]

condition = []
for i in range(1, len(choose_point)):
    pair_use = []
    for j in range(i + 1):
        if j == 0:
            pair = [pair_all[i], nneigh[pair_all[i]]]
            pair_use.append(nneigh[pair_all[i]])
        else:
            if pair_all[j - 1] in pair_use:
                continue
            else:
                pair = [pair_all[i], pair_all[j - 1]]
                pair_use.append(pair_all[j - 1])

        dist_ther1 = dist[pair[0], pair[1]]
        dist_ther = dist_ther1 * dist_ther1_ratio
        turn, nei_all, find_time, count, nei_dist = judge_nei(pair, dist, count, dc, dist_ther)

        if turn == 2:
            punish_time = find_time - 5
            dist_new = nei_dist + dist_ther * (1 + punish_time * punish_ratio)
            dist[pair[0], pair[1]] = dist_new
            dist[pair[1], pair[0]] = dist_new
            if dist_new < delta[pair_all[i]]:
                delta[pair_all[i]] = dist_new
                nneigh[pair_all[i]] = pair[1]

        if turn == 3 and j == 0:
            max_dist_nei = np.max(dist[pair[1], nei_all])
            dist[pair[0], pair[1]] = max_dist_nei * 1.1
            dist[pair[1], pair[0]] = dist[pair[0], pair[1]]
            delta[pair_all[i]] = dist[pair[0], pair[1]]

    condition1 = [pair, turn, find_time, nei_dist, dist_ther1, dist[pair[0], pair[1]]]
    condition.append(condition1)

maxd = np.max(dist)
delta[ordrho[0]] = np.max(delta)

#rhomin, deltamin, NCLUST = decision_graph(rho, delta)

rhomin, deltamin = np.min(rho), np.min(delta)

cl, icl, NCLUST = find_cluster_centers(rho, delta, rhomin, deltamin)

cl, halo = perform_assignation(rho, delta, ordrho, nneigh, dist, dc, NCLUST, icl)
"""
# Final Plot :
if len(data[0,:])>2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0],data[:, 1],data[:, 2], c=cl, alpha=1, s=10)
else:
    plt.scatter(data[:, 0],data[:, 1], c=cl, alpha=1, s=20)

# Calculate FMI
#DPCFMI = fowlkes_mallows_score(true_label, cl)
#print(f'FMI value of DPC on fourlines dataset: {DPCFMI}')

# Calculate ARI
DPCARI = adjusted_rand_score(true_label, cl)
print(f'ARI value of DPC on fourlines dataset: {DPCARI}')

# Calculate NMI
DPCNMI = normalized_mutual_info_score(true_label, cl)
print(f'NMI value of DPC on fourlines dataset: {DPCNMI}')
"""