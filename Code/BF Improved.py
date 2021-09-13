import random
from threading import Thread
import numba 
import numpy
from sklearn.cluster import DBSCAN

data=-1		# dummy value
clusters=-1	# dummy value
matrix=-1	# dummy value
n_centroid=0
centroid_impacts_sample=[]

#@numba.njit()
def init_dist_matrix(centroids):
	for instance in range(len(data)):
		for centroid in centroids:
			matrix[instance].append(numpy.linalg.norm(data[instance] - centroid))

def retrievecentroids(sample):
	# Sampling can cause issues if abnormal samples are choses. This may lead to more clusters than desired (Ill effect of DBSCAN)
	# Computes centroids
	dbscan=DBSCAN(0.5,10)
	# Can cause issues because we don't know the parameters for dbscan
	dbscan.fit(sample)
	centroids=[[] for _ in range(len(numpy.unique(dbscan.labels_))-1)]
	for instance in range(len(dbscan.labels_)):
		centroids[dbscan.labels_[instance]].append(numpy.array(sample[instance]))

	global centroid_impacts_sample
	centroid_impacts_sample=[len(centroid) for centroid in centroids]

	for centroid in range(len(centroids)):
		centroids[centroid]=numpy.mean(centroids[centroid],axis=0)
	print('centroids',centroids)
	return centroids

def retrievethresholds(centroids):
	# Computes max-distance using centroids
	# Average threshold - Method #1
	avg_t=0
	for centroid in centroids:
		for secondcentroid in centroids:
			avg_t+=numpy.linalg.norm(secondcentroid - centroid)
	if len(centroids)!=1:
		avg_t/=len(centroids)*(len(centroids)-1)
	print('avg threshold',avg_t)
	thresholds=[avg_t]*len(centroids)

	for threshold in range(len(thresholds)):
		thresholds[threshold]=thresholds[threshold]*centroid_impacts_sample[threshold]/sum(centroid_impacts_sample)

	return thresholds

class Cluster(Thread):
	def __init__(self,threshold):
		Thread.__init__(self)
		global n_centroid
		self.centroid=n_centroid
		n_centroid+=1
		self.threshold=threshold
	def run(self):
		# Clustering starts here
		while(True):
			if(len(matrix)==0):
				break
			min_d=matrix[0][self.centroid]
			min_d_index=0
			for i in range(len(matrix)):
				if(matrix[i][self.centroid]<min_d):
					min_d_index=i
					min_d=matrix[i][self.centroid]
			if(min_d<self.threshold):
				clusters[self.centroid].append(data[min_d_index])
				del data[min_d_index]
				del matrix[min_d_index]
			else:
				break
		print('Cluster ',len(clusters[self.centroid]))

def sampling():
	returnable={}
	sample=random.sample(list(data),round(0.2*len(data)))
	centroids=retrievecentroids(sample)
	returnable['centroids']=centroids
	global clusters
	clusters=[[] for _ in range(len(centroids))]
	thresholds=retrievethresholds(centroids)
	returnable['thresholds']=thresholds
	return returnable

def bf(parameters):
	centroids=parameters['centroids']
	thresholds=parameters['thresholds']
	init_dist_matrix(centroids)
	cluster=[[] for _ in range(len(centroids))]
	for iterator in range(len(centroids)):
		cluster[iterator]=Cluster(thresholds[iterator])
		cluster[iterator].setName('Cluster '+str(iterator))
	for iterator in range(len(centroids)):
		cluster[iterator].start()
	for iterator in range(len(centroids)):
		cluster[iterator].join() 

from matplotlib import pyplot as pl 
from sklearn.decomposition import PCA

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

import csv
if __name__ == '__main__':
	# Load data
	with open('brambles.csv','r') as f:
		reader=csv.reader(f)
		df=list(reader)
	df=[[float(j) for j in i] for i in df]

	from copy import deepcopy
	data=deepcopy(df)


	pca = PCA(n_components=2).fit(df)
	pca_2d = pca.transform(df)
	

	matrix=[[] for _ in range(len(df))]
	parameters=sampling()
	bf(parameters)
	print('noise instances = ',len(data))
	print('n_clusters',len(clusters))

	for i in range(pca_2d.shape[0]):
		if compare(df[i],clusters[0]):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
		if(len(clusters)>=2):
			if compare(df[i],clusters[1]):
				pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='*')
		if(len(clusters)>=3):
			if compare(df[i],clusters[2]):
				pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='.')
	'''
	# For noise
	pca = PCA(n_components=2).fit(data)
	pca_2d = pca.transform(data)
	for i in range(pca_2d.shape[0]):
		pl.scatter(pca_2d[i,0],pca_2d[i,1],c='k',marker='+')
'''
	pl.show()