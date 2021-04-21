# TSGEM_Clustering
This repository is designed to cluster time-series Gene Expreession Matrix (GEM) based on the gene trajectory pattern.
## Motivation
As most biological processes are dynamic, time-series transcriptome experiments play a pivotal role in understanding and modeling these processes. Using K-Means to profiling the time-course transcriptional response is a common approach for bioinfomaticians. TSGEM_Clustering will optimaize the KMeans performance on sequential time-series data by adapting Dynamic Time Warping Distance Metric. And it will select the optimal number of clusters based on the results from distortion (also known as elbow method),silhouette coefficient, and Calinski harabasz index. 
## Installation
All of TSGEM_Clustering's dependencies can be installed through Anaconda3. To create an Anaconda environment:
```
#Specific to Clemson's Palmetto Cluster
module load anaconda3/5.1.0-gcc/8.3.1

conda create -n TSGEM_Clustering python=3.6 matplotlib numpy pandas scikit-learn 
```
Once the anaconda environment has been created, the ```tslearn``` package must be installed seperately
```
source activate TSGEM_Clustering
conda install -c conda-forge tslearn
```
## Test

### Step1: Choose the Optimal K
```
python DTW-perf.py -i Test/test.txt -kmin 2 -kmax 30 -step 2 -o Test_Results/Step1Test
```
### Step2: DTW-KMeans Clustering
```
python DTW_argv.py -i Test/test.txt -k 6 -o Test_Results/Step2Clustering -p Test-K6
```
