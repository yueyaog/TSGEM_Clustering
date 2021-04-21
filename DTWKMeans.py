#!/home/yueyaog/.conda/envs/deep-learning/bin/python3
########################################################################################
#
# DTWKmeans.py
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
from tslearn.clustering import TimeSeriesKMeans
import argparse
########################################################################################
#
# Description of script
#
########################################################################################

parser = argparse.ArgumentParser(description="""
DTWKMneans.py takes a gene expression matrix as input and run soft-DTW k-means 
according to Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)

This script returns optimal clustering results and gene expression clustering plots
that can be used for downstream analysis
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
parser.add_argument('-k','--K',dest="optimal_K",type=int,action='store',help="The optimal number of K")
parser.add_argument('-o','--output',dest="output_path_dir",action='store')
parser.add_argument('-p','--prefix',dest="output_file_prefix",action='store')
args = parser.parse_args()
#########################################################################################

##########################################################################################
exp_df = pd.read_csv(args.gene_expression_matrix,sep="\t",index_col='gene')
MTRinput_df = pd.read_csv(args.gene_expression_matrix,sep='\t',index_col='gene')
MTR_arr = MTRinput_df.values
print('Input Gene Expression Matrix has {} entries with {} time points'.format(MTR_arr.shape[0],MTR_arr.shape[1]))
# Normalization with standard scaler
gene_expression_matrix = MTR_arr
gene_expression_matrix -= np.vstack(np.nanmean(gene_expression_matrix, axis=1))
gene_expression_matrix /= np.vstack(np.nanstd(gene_expression_matrix, axis=1))

# soft-DTW-Kmeans
# UPDATES(FEB19-2021), implant the argv for optimal K
# seed of 10 for reproducibility 
dtw_Ymtr = TimeSeriesKMeans(n_clusters=args.optimal_K, metric="softdtw", metric_params={"gamma": .01}, verbose=True,random_state=10,n_jobs=-1)
Ymtr_predict = dtw_Ymtr.fit_predict(gene_expression_matrix)
print('The Shape of Cluster Centers are {}'.format(dtw_Ymtr.cluster_centers_.shape))

# CREATE A DIRECTORY TO SAVE THE RESULTS
os.mkdir(args.output_path_dir)
os.chdir(args.output_path_dir)

# Print out the cluster result output
df_label = pd.DataFrame()
df_label['gene'] = MTRinput_df.index
df_label['cluster'] = dtw_Ymtr.labels_+1
df_label.to_csv(args.output_file_prefix+"_ClusteringResults.csv")

# Print out the result summary and save it as a csv
df_sum = pd.DataFrame()
cluster_list = []
GeneNum_list = []
for i,j in collections.Counter(Ymtr_predict).items():
    cluster_list.append(i+1)
    GeneNum_list.append(j)

    print("Cluster {} : {}".format(i+1,j))
df_sum['Cluster'] = cluster_list
df_sum['GeneNum'] = GeneNum_list
df_sum.to_csv(args.output_file_prefix+"_ClusteringSummary.csv")


# Time Set for plotting
t_label = list(map(int,MTRinput_df.columns.tolist()))
t = t_label
t /= np.mean(np.diff(t))

# Plot
os.chdir(args.output_path_dir)
plt.figure(figsize=(20,20))
# The cols and rows of the subplot 
subplots = int(math.sqrt(args.optimal_K))+1
for yi in range(args.optimal_K):
    ax = plt.subplot(subplots,subplots, yi + 1)
    for xx in gene_expression_matrix[Ymtr_predict == yi]:
        ax.plot(t,xx.ravel(),'k-', alpha=0.15)
    ax.plot(t,dtw_Ymtr.cluster_centers_[yi].ravel(), "r-")
    ax.set_xticks(t)
    ax.set_xticklabels(t_label)
    ax.set_ylabel('Gene expression')
    ax.set_title('Cluster {} ({})'.format(yi + 1,collections.Counter(Ymtr_predict)[yi]))
    ax.grid(False)
    if yi == 1:
        plt.title("DTW Mtr $k$-means")
        
plt.tight_layout()
plt.savefig(args.output_file_prefix+'_clustering.png',dpi=200)
