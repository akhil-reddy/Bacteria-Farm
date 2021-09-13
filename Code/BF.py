import random
import time
from threading import Thread
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
matrix=-1	# dummy value
n_centroid=0
labels_=[]	# dummy value	

#@numba.njit()
def init_dist_matrix(centroids):
	for instance in range(len(data)):
		for centroid in centroids:
			matrix[instance].append(numpy.linalg.norm(data[instance] - centroid))

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
		thresholdn[i]=len(centroids[i])/sumn
		centroids[i]=numpy.mean(centroids[i],axis=0)
	print('centroids',centroids)
	print('thresholdn',thresholdn)
	print(numpy.unique(kmeans.labels_))
	return centroids,thresholdn

class Cluster(Thread):
	def __init__(self,thresholdn):
		Thread.__init__(self)
		global n_centroid
		self.centroid=n_centroid
		n_centroid+=1
		self.thresholdn=round(thresholdn*len(data))
	def run(self):
		# Clustering starts here
		while(True):
			if(len(matrix)==0):
				break
			if(matrix[0]!=-1):
				min_d=matrix[0][self.centroid]
			else:
				for _ in matrix:
					if(_!=-1):
						min_d=_[self.centroid]
						break
			min_d_index=0
			for i in range(len(matrix)):
				if(matrix[i]!=-1 and matrix[i][self.centroid]<min_d):
					min_d_index=i
					min_d=matrix[i][self.centroid]
			if(len(clusters[self.centroid])<self.thresholdn and data[min_d_index]!=-1):
				clusters[self.centroid].append(data[min_d_index])
				data[min_d_index]=-1
				matrix[min_d_index]=-1
				labels_[min_d_index]=self.centroid
			else:
				break
		print('Cluster ',len(clusters[self.centroid]))

def sampling(n):
	returnable={}
	sample=random.sample((data),round(0.2*len(data)))
	centroids,thresholdn=retrievecentroidsandthresholdn(sample,n)
	returnable['centroids']=centroids
	global clusters
	clusters=[[] for _ in range(len(centroids))]
	returnable['thresholdn']=thresholdn
	return returnable

def bf(parameters):
	centroids=parameters['centroids']
	thresholdn=parameters['thresholdn']
	global labels_
	labels_=[-1 for _ in range(len(data))]
	init_dist_matrix(centroids)
	cluster=[[] for _ in range(len(centroids))]
	for iterator in range(len(centroids)):
		cluster[iterator]=Cluster(thresholdn[iterator])
		cluster[iterator].setName('Cluster '+str(iterator))
	for iterator in range(len(centroids)):
		cluster[iterator].start()
	for iterator in range(len(centroids)):
		cluster[iterator].join() 

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

	pca = PCA(n_components=2).fit(df)
	pca_2d = pca.transform(df)
	
	start=time.time()
	matrix=[[] for _ in range(len(df))]
	parameters=sampling(int(sys.argv[2]))
	bf(parameters)
	elapsed=time.time()-start

	print('time =',elapsed)
	print('Silhouette Coefficient =',metrics.silhouette_score(df,labels_,metric='euclidean'))
	print('Calinski-Harabaz Index =',metrics.calinski_harabaz_score(df,labels_))
	print('n_clusters',len(clusters))

	for i in range(pca_2d.shape[0]):
		if(labels_[i]==0):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
		elif(labels_[i]==1):
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='+')
		else:
			pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='+')
	pl.axis('equal')
	pl.show()