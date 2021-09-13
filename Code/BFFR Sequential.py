import random
import time
import numba 
import numpy
import sys
from sklearn.cluster import KMeans
from matplotlib import pyplot as pl 
from sklearn.decomposition import PCA
import csv
from sklearn import metrics

data=-1		# dummy value
clusters=-1	# dummy value
n_centroid=0
frs=-1		# dummy value
labels_=[]	# dummy value	
noise=-1	# dummy value

def sumcn(centroids):
	s=0
	for centroid in centroids:
		s+=len(centroid)
	return s

def retrievecentroidsandthresholdn(sample,n):
	# Computes centroids and proportion of data for each centroid
	kmeans=KMeans(n_clusters=n)
	kmeans.fit(sample)
	centroids=[[] for _ in range(len(numpy.unique(kmeans.labels_)))]
	for i in range(len(kmeans.labels_)):
		if kmeans.labels_[i]==-1:
			continue
		centroids[kmeans.labels_[i]].append(numpy.array(sample[i]))
	sumn=sumcn(centroids)
	thresholdn=[-1 for _ in range(len(centroids))]
	for i in range(len(centroids)):
		thresholdn[i]=len(centroids[i])/sumn*(1-noise)		# Specify the noise
		centroids[i]=numpy.mean(centroids[i],axis=0)
	print('centroids',centroids)
	print('thresholdn',thresholdn)
	return centroids,thresholdn

def sampling(n,fr):
	# First part of the modular algorithm
	returnable={}
	sample=random.sample((data),round(0.2*len(data)))
	centroids,thresholdn=retrievecentroidsandthresholdn(sample,n)
	returnable['centroids']=centroids
	returnable['thresholdn']=thresholdn
	returnable['front runners']=[[-1 for __ in range(fr)] for _ in range(len(centroids))]
	global clusters
	clusters=[[] for _ in range(len(centroids))]
	return returnable

def bf(parameters):
	centroids=parameters['centroids']
	global labels_
	labels_=[-1 for _ in range(len(data))]
	cluster=[[] for _ in range(len(centroids))]
	thresholdn=parameters['thresholdn']
	global frs
	frs=parameters['front runners']

	# Second part of the algorithm
	for iterator in range(len(centroids)):
		global n_centroid
		centroid=n_centroid
		fr=frs[n_centroid]
		fr[0]=centroids[centroid]
		n_centroid+=1
		threshold=round(thresholdn[iterator]*len(data))

		# Clustering starts here
		while(True):
			if(len(clusters[centroid])>=threshold):
				break
			min_instance=-1				# dummy value
			min_d=-1					# dummy value
			global_min_instance=-1		# dummy value
			global_min_d=99999999		# dummy value
			global_min_index=-1			# dummy value
			for instance in range(len(data)):
				if(labels_[instance]!=-1):
					continue
				min_ds=[]
				for runner in range(len(fr)):
					if(type(fr[runner]) is numpy.ndarray):
						min_ds.append(numpy.linalg.norm(data[instance]-fr[runner]))
					else:
						break
				min_d=min(min_ds)
				min_instance=data[instance]
				if(min_d<global_min_d):
					global_min_index=instance
					global_min_instance=min_instance
					min_instance=-1
					global_min_d=min_d
					min_d=-1
			clusters[centroid].append(global_min_instance)
			for i in range(len(fr)):	
				if(type(fr[i]) is not numpy.ndarray): 	# appending procedure
					fr[i]=global_min_instance
					labels_[global_min_index]=centroid
					break
			else:										# replacement procedure
				fr[min_ds.index(min(min_ds))]=global_min_instance
				labels_[global_min_index]=centroid
		print('Clustered ',len(clusters[centroid]))

def compare(instance,labels):
	ret=False
	for i in labels:
		for value in range(len(i)):
			ret=True
			if(i[value]!=instance[value]):
				ret=False
		if(ret==True):
			return ret
		ret=False
	return ret

if __name__ == '__main__':
	# Load data
	with open(sys.argv[1],'r') as f:
		reader=csv.reader(f)
		df=list(reader)
	df=[[float(j) for j in i] for i in df]

	from copy import deepcopy
	data=deepcopy(df)
	noise=float(sys.argv[4])
	pca = PCA(n_components=2).fit(df)
	pca_2d = pca.transform(df)
	
	start=time.time()
	matrix=[[] for _ in range(len(df))]
	parameters=sampling(int(sys.argv[2]),int(sys.argv[3]))
	bf(parameters)
	elapsed=time.time()-start

	print('time =',elapsed)
	print('Silhouette Coefficient =',metrics.silhouette_score(df,labels_,metric='euclidean'))
	print('Calinski-Harabaz Index =',metrics.calinski_harabaz_score(df,labels_))
	print('n_clusters',len(clusters))

	print(numpy.unique(labels_))
	for i in range(pca_2d.shape[0]):
		if(labels_[i]==0):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
		elif(labels_[i]==1):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='+')
		elif(labels_[i]==2):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='+')
		else:
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='+')
	pl.axis('equal')
	pl.show()