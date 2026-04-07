#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import munkres
import numpy as np
from sklearn.mixture import GaussianMixture
def kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Kullback-Leibler divergence between two Gaussian distributions.
    
    Parameters:
        mu1 (float): Mean of the first Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
        
    Returns:
        kl_div (float): Kullback-Leibler divergence.
    """
    #kl_div = np.linalg.norm(np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)
    kl_div = np.mean(np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)
    return kl_div
def Bucketization(cluster1,cluster2,indexes,turn,centers):
	if turn==0:
		for i in range(len(cluster1)):
			centers[i].append(cluster1[i])
	  
	
	for i in range(len(cluster2)):
		r,c=indexes[i]
		centers[r].append(cluster2[c])

	return centers



"""
Matching between set of centroid. This function is called when T_E <= 5
"""
def weightMatrix(clusters,varince, dd_list,k_centers):
	count=0
	avg_score=0
	K=len(clusters[0])
	for i in range(len(clusters)):
		cluster1=clusters[i]
		varince1=varince[i]
		for j in range(i+1,len(clusters)):
			cluster2=clusters[j]
			varince2=varince[j]
			count=count+1
			weight=np.zeros([K,K])
			weight_bk=np.zeros([K,K])
			#~~~~~computation of distance between two sets of centroids~~~~~~~
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				vec3=np.asarray(varince1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])
					vec4=np.asarray(varince2[n])
					#weight_bk[m][n]=np.linalg.norm(vec1-vec2)+np.linalg.norm(vec3-vec4)*0.1
					#weight[m][n]=np.linalg.norm(vec1-vec2)+np.linalg.norm(vec3-vec4)*0.1
					weight_bk[m][n] = kl_divergence(vec1, vec3, vec2, vec4)
					weight[m][n] = kl_divergence(vec1, vec3, vec2, vec4)
					
			score=0
			#~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)

			#~~~~~Similar centroids are put in same bucket~~~~~~
			if i==0:
				Bucketization(cluster1,cluster2,indexes,j,k_centers)

			#~~~~~~CNAK score~~~~~~~~~
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			avg_score=avg_score+score
			dd_list.append(score)
	
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers


"""
Matching between set of centroid. This function is called when T_E < T_threshold
"""

def weightMatrixUpdated(global_centroids_list,clusters,global_varince_list,varince,dd_list,k_centers,avg_score,count):
	
	avg_score=avg_score*count

	K=len(clusters[0])
	
	for i in range(len(global_centroids_list)):
		cluster1=global_centroids_list[i]
		varince1=global_varince_list[i]
		for j in range(len(clusters)):
			cluster2=clusters[j]
			varince2=varince[j]
			count=count+1
			weight=np.zeros([K,K])
			weight_bk=np.zeros([K,K])
			#~~~~~computation of distance between two sets of centroids~~~~~~~	
			for m in range(len(cluster1)):
					
				vec1=np.asarray(cluster1[m])
				vec3=np.asarray(varince1[m])
				for n in range(len(cluster2)):
					vec2=np.asarray(cluster2[n])
					vec4=np.asarray(varince2[n])
					#weight_bk[m][n]=np.linalg.norm(vec1-vec2)+np.linalg.norm(vec3-vec4)*0.1
					#weight[m][n]=np.linalg.norm(vec1-vec2)+np.linalg.norm(vec3-vec4)*0.1
					weight_bk[m][n] = kl_divergence(vec1, vec3, vec2, vec4)
					weight[m][n] = kl_divergence(vec1, vec3, vec2, vec4)
					
			score=0
			#~~~~~Computation of perfect matching between two sets of centroids using Kuhn-Munkre's Algorithm~~~~~
			matching = munkres.Munkres()
			indexes = matching.compute(weight_bk)
			#~~~~~Similar centroids are put in same bucket~~~~~~
			if i==0:
				Bucketization(cluster1,cluster2,indexes,j,k_centers)
			#~~~~~~CNAK score~~~~~~~~~
			for r, c in indexes:
				score=score+weight[r][c]
			score=score/len(cluster1)
			dd_list.append(score)
			avg_score=avg_score+score
	avg_score=avg_score/count
	return avg_score, count, dd_list, k_centers


"""
Core function of CNAK
"""

def CNAK_core(data,gamma,K):

	T_S=1
	T_E=5
	
	centroids_list=[]
	varince_list = []
	scores_list=[]
	for j in range(T_S,T_E):
		#~~~~~random sampling without repitiion~~~~~~~~
		#gamma1=0.15*j
		index=random.sample(range(0, len(data)),int(len(data)*gamma))
		samples=[]
		for k in range(int(len(data)*gamma)):
			temp=data[index[k]]
			samples.append(temp)
		#~~~~~~K-means++ on sampled dataset~~~~~~~~~
		kmeans = GaussianMixture(n_components=K, tol=0.001, n_init=20, init_params='k-means++',covariance_type='diag').fit(samples)
		scores_list.append(kmeans.score(np.array(samples)))
		centroids=kmeans.means_
		varince= kmeans.covariances_
		centroids_list.append(centroids)	
		varince_list.append(varince)
	dd_list=[]
	k_centers=[[] for i in range(len(centroids))]
	#~~~~~~Computation of CNAK score and forming K buckets with T_E similar  centroids~~~~~~~~
	avg_score, count, dd_list,k_centers=weightMatrix(centroids_list,varince_list, dd_list,k_centers)
	
	#~~~~~Estimate the value of T ~~~~~~~~~
	mean=np.mean(dd_list)
	std=np.std(dd_list)
	val=(1.414*20*std)/(mean)
	
	global_centroids_list=[]
	for centroids in (centroids_list):
		global_centroids_list.append(centroids)
	centers=[[] for i in range(len(centroids_list[0]))]

	global_varince_list=[]
	for varince in (varince_list):
		global_varince_list.append(varince)
	vari=[[] for i in range(len(centroids_list[0]))]
	
	#~~~~~Repeat untill  T_E > T_threshold ~~~~~~~~~
	while val>T_E:
		T_S=T_E
		T_E=T_E+1
		centroids_list=[]
		varince_list = []
		for j in range(T_S,T_E):
			index=random.sample(range(0, len(data)),int(len(data)*gamma))
			datax=[]
			
			for k in range(int(len(data)*gamma)):
				temp=data[index[k]]
				datax.append(temp)
			
			kmeans = GaussianMixture(n_components=K, tol=0.001, n_init=20, init_params='k-means++',covariance_type='diag').fit(datax)
			scores_list.append(kmeans.score(np.array(datax)))
			varince= kmeans.covariances_
			centroids_list.append(centroids)	
			varince_list.append(varince)
	
		avg_score, count, dd_list,k_centers=weightMatrixUpdated(global_centroids_list,centroids_list,global_varince_list,varince_list,dd_list,k_centers,avg_score,count)
		for centroids in ((centroids_list)):
			global_centroids_list.append(centroids)
		for varince in (varince_list):
			global_varince_list.append(varince)
		mean=np.mean(dd_list)
		std=np.std(dd_list)
		val=(1.414*20*std)/(mean)
		if T_E>15:
			break
	bic=sum(scores_list)/len(scores_list)
	clusterCenterAverage=[]
	for i in range(len(k_centers)):
		  clusterCenterAverage.append(np.mean(k_centers[i],axis=0))
	return val, T_E, avg_score, clusterCenterAverage,bic


"""
Generating cluster Label for K_hat
"""

def LabelGeneration(data,k_centers,method):
	

	kmeans = GaussianMixture(n_components=k_centers, max_iter=300,tol=0.0001, n_init=20, init_params='k-means++').fit(data)
	clusterLabel=kmeans.predict(data)
	file=open("CNAK_labels"+"_"+method+".txt","a")

	for i in range(len(clusterLabel)):
		file.write(str(clusterLabel[i]))
		file.write(str("\n"))
	file.close()
	

	return 

"""
CNAK Implementation
gamma and k_max are optional paraneters. The heuristic used in CNAK paper, can be used for computing gamma. 
"""


#def CNAK(data, gamma=0.7, k_min=1, k_max=21):
def CNAK(data, method, alpha,**kwargs):
		
	if kwargs['gamma']==None:
		gamma=0.7
	else:
		gamma=kwargs['gamma']

	if kwargs['K_min']==None:
		k_min=1
	else:
		k_min=kwargs['K_min']

	if kwargs['K_max']==None:
		k_max=21
	else:
		k_max=kwargs['K_max']
	print(" gamma:",gamma," K_min:",k_min," K_max:",k_max)
	CNAK_score=[]
	k_max_centers=[]
	file=open("CNAK_scores"+"_"+method+".csv","a")
	for K in range(k_min,k_max):
		

		val, T_E, avg_score, k_centers,bic=CNAK_core(data,gamma,K)
		if K==k_min:
			avg_score_stand=avg_score
			bic_stand=bic
		avg_score=avg_score/avg_score_stand
		bic=bic#/bic_stand
		avg_score=bic#+alpha*avg_score
		CNAK_score.append(avg_score)
		k_max_centers.append(k_centers)
		file.write(str(K))
		file.write(str(","))
		file.write(str(avg_score))
		file.write(str("\n"))
		print(K)
		
	file.close()
	K_hat=caculate_point_change(CNAK_score)#CNAK_score.index(min(CNAK_score))
	print("K_hat:",K_hat+k_min)
	#~~~~~~~~Labels for K_hat~~~~~~~~~
	return K_hat+k_min
    
def caculate_point_change(points):
    slopes = []
    for i in range(len(points) - 1):
        y1 = points[i]
        y2 = points[i + 1]
        slope = (y2 - y1) / 1
        slopes.append(slope)
    differences = []
    for i in range(len(slopes) - 1):
        difference = slopes[i + 1] - slopes[i]
        differences.append(difference)
    return differences.index(max(differences))+1

from collections import Counter

def find_most_frequent(lst):
    counts = {}
    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    max_count = max(counts.values())
    most_common_items = [item for item, count in counts.items() if count == max_count]
    return min(most_common_items)

def process_labels(lst,cn):
    counter=Counter(lst)
    temp=[]
    for i in range(cn):
        temp.append(counter.most_common(i+1)[i][0])
    for i in range(len(lst)):
        if lst[i] in temp:
            continue
        else:
            lst[i]=-1
    return lst