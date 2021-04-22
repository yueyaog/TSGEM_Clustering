#!/usr/bin/env python
########################################################################################
#
# DTW-perf.py
# Author: Gao Yueyao
# Python 3.6.10
# Requires the following Python packages:
# numpy(=1.18.1), pandas(1.0.3), matplotlib(3.2.1), tslearn(=0.41)
#
########################################################################################
#
# Import depenencies
#
########################################################################################
import os
import math
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from tslearn.clustering import TimeSeriesKMeans
import argparse
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
########################################################################################
#
# Description of script
#
########################################################################################
parser = argparse.ArgumentParser(description="""
DTW-perf.py takes a gene expression matrix as input and run soft-DTW k-means 
according to Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541).

The purpose of this script is to select the optimal number of clusters by 
evaluating the performance of the DTW-KMeans clustering algorithm by 
fitting the model with a range of K values.
This script uses three performance evaluation method: 
1. Elbow Method
    Calculate Distortion (the average of the squared distances from the cluster centers of the respective clusters)

2. Silhouette coefficient[1]:
    near +1 indicate that the sample is far away from the neighboring clusters
    https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

3. Calinski harabasz Index[2]: 
    Computes the ratio of dispersion between and within clusters.
    A higher Calinski-Harabasz score relates to a model with better defined clusters.

[1]Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis”. 
Computational and Applied Mathematics 20: 53–65. doi:10.1016/0377-0427(87)90125-7.
[2]Caliński, T., & Harabasz, J. (1974). “A Dendrite Method for Cluster Analysis”. 
Communications in Statistics-theory and Methods 3: 1-27. doi:10.1080/03610927408827101.
""")

#########################################################################################
#
# Required arguments
#
##########################################################################################
parser.add_argument('-i','--input',dest='gene_expression_matrix',action="store",required=True,help="""
The format of the gene_expression_matrix.txt is:
gene    1     2    3    ...    time_t
gene_1  10    20   5    ...    8
gene_2  3     2    50   ...    8
gene_3  18    100  10   ...    22
...
gene_n  45    22   15   ...    60
""")
parser.add_argument('-kmin','--Kmin',dest="Kmin",type=int,action='store',help="The minimum K value")
parser.add_argument('-kmax','--Kmax',dest="Kmax",type=int,action='store',help="The maximum K value")
parser.add_argument('-step','--stepsize',dest="stepsize",type=int,action='store',help="step size of K value range")
parser.add_argument('-o','--output',dest="output_path_prefix",action='store',help='required, e.g. /path/to/PERF_results')
args = parser.parse_args()
##########################################################################################
exp_df = pd.read_csv(args.gene_expression_matrix,sep="\t",index_col='gene')
MTRinput_df = pd.read_csv(args.gene_expression_matrix,sep='\t',index_col='gene')
MTR_arr = MTRinput_df.values
print('Input Gene Expression Matrix has {} entries with {} time points'.format(MTR_arr.shape[0],MTR_arr.shape[1]))
# Normalization with standard scaler
gene_expression_matrix = MTR_arr
gene_expression_matrix -= np.vstack(np.nanmean(gene_expression_matrix, axis=1))
gene_expression_matrix /= np.vstack(np.nanstd(gene_expression_matrix, axis=1))

# Set up a K range to iterate through for DTW-KMeans model
K_range = np.arange(args.Kmin,args.Kmax,args.stepsize)
# Three ways to measure the performances of DTW-KMeans model 
Sum_of_squared_distances = []
ch_indexs = []
silhouette_scores = []
for n_clusters in K_range:
    # soft-DTW-Kmeans
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility 
    clusterer = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", metric_params={"gamma": .01}, verbose=False,random_state=10)
    cluster_labels = clusterer.fit_predict(gene_expression_matrix)
    print('The Shape of Cluster Centers are {}'.format(clusterer.cluster_centers_.shape))
    # The squared distance for Elbow Method
    # Select optimal number of clusters by fitting the model 
    # with a range of K values 
    Sum_of_squared_distances.append(clusterer.inertia_)
    print("For n_clusters =", n_clusters,
          "The sum of squared distance is :", clusterer.inertia_)
    #Compute the Calinski and Harabasz score.
    #This gives ratio between the within-cluster dispersion and the 
    # between-cluster dispersion.
    ch_score = calinski_harabasz_score(gene_expression_matrix, cluster_labels)
    ch_indexs.append(ch_score)
    print("For n_clusters =", n_clusters,
          "The calinski_harabasz_score is :", ch_score)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(gene_expression_matrix, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(gene_expression_matrix, cluster_labels)
    # Plot the actual time-series clusters formed
    # Time Set 
    t_label = list(map(int,MTRinput_df.columns.tolist()))
    t = t_label
    t /= np.mean(np.diff(t))
    plt.figure(figsize=(20,20))
    subplots = int(math.sqrt(n_clusters))+1
    for yi in range(n_clusters):
        ax = plt.subplot(subplots,subplots, yi + 1)
        for xx in gene_expression_matrix[cluster_labels == yi]:
            ax.plot(t,xx.ravel(),'k-', alpha=0.15)
        ax.plot(t,clusterer.cluster_centers_[yi].ravel(), "r-")
        ax.set_xticks(t)
        ax.set_xticklabels(t_label)
        ax.set_ylabel('Gene expression')
        ax.set_title('Cluster {} ({})'.format(yi + 1,collections.Counter(cluster_labels)[yi]))
        ax.grid(False)
        if yi == 1:
            plt.title("DTW Mtr $k$-means")
    plt.tight_layout()
    plt.savefig(args.output_path_prefix+'K{}_clustering.png'.format(n_clusters),dpi=200)
#Plot the results of Elbow Method, silhouette score, and ch index
fig, (ax0, ax1, ax2) = plt.subplots(3,figsize=(7, 21))
ax0.plot(K_range, Sum_of_squared_distances, 'bo-')
ax0.set_xlabel('n_clusters')
ax0.set_ylabel('Sum of Squared Distances')
ax0.set_title('The Elbow Method using Distortion for DTW-KMeans Clustering')
ax0.grid(False)

ax1.plot(K_range, ch_indexs, 'g.-')
ax1.set_xlabel('n_clusters')
ax1.set_ylabel('ch_score')
ax1.set_title('Calinski and Harabaz Score for DTW-KMeans Clustering')
ax1.grid(False)

ax2.plot(K_range, silhouette_scores, 'r*-')
ax2.set_xlabel('n_clusters')
ax2.set_ylabel('Averge Silhouette Score')
ax2.set_title('Averge Silhouette Score for DTW-KMeans Clustering')
ax2.grid(False)

plt.tight_layout()
plt.savefig(args.output_path_prefix+'_DTW-KMeans_PERFs.png',dpi=200)