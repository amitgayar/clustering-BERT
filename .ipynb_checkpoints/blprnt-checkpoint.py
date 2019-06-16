# paragraph :
def art_to_para(article):
	para_list = article.split('\n\n')
	l = len(para_list)
	def para_thresholding(i=0, clustered_para=[]):
		temp = para_list[i]
		# print (i,' first')
		while len(temp.split(" ")) < threshold and i < l - 1:
			temp = temp + " " + para_list[i+1]
			i+=1
			# print("while ",i)
		clustered_para.append(temp) 
		# print(clustered_para)
		if i == l-1:
			return clustered_para
		else:
			return para_thresholding(i+1,clustered_para)

	return para_thresholding()

def tfidf_pruning():
    idf.drop values if value < threshold 
    keys = filter(lambda x: (idf[x] <threshold), idf.keys())
    
    for dict in tfdoc:
        lambda x: del(key), keys
----------------

processed_doc = pickle.load(open('spacy_cleansing.pkl','rb'))
wordSet = processed_doc['wordSet']
doc = processed_doc['clean_text_doclist']
para_count = processed_doc['para_count_in_art']

art_index = []
temp = 0
for p in para_count:
    temp += p
    art_index.append(temp)


doc_set_all = []
for docc in doc:
	doc_set = sorted(set().union(docc))
	doc_dict = dict.fromkeys(doc_set, 0)
	for word in docc:
		doc_dict[word] += 1 
	doc_set_all.append(doc_dict.copy())

# print ("wordSet  :   "," ".join(sorted(wordSet)))

def computeTF(tfDict,doc):
    doc_count = len(doc)
    for word, count in tfDict.items():
    	tfDict[word] = count/doc_count
    return tfDict

tfdoc = []
for i in range(len(doc)):
    tfdoc.append(computeTF(doc_set_all[i], doc[i]))
from functools import reduce

i=0
art_word_count = []
art_index = [0] + art_index
while i<len(art_index)-1:
    doc_temp = doc[art_index[i]:art_index[i+1]]
    i+=1
    temp = 0
    for d in doc_temp:
        temp +=len(d)
    art_word_count.append(temp)
    
    ----------------------------------k mean clustering--


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm
from copy import deepcopy

dataset=pd.read_csv('tfidf_of_paras.csv')
X = dataset.iloc[:,1:].values
X = np.nan_to_num(X)
# print(X)
K=5
m=X.shape[0]
feat = X.shape[1]
Centroids=np.array([]).reshape(feat,0)

# print(Centroids)



# --------second time implementation-----------------
# https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python
mean = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
# print(mean,std)
centers = np.random.randn(K,feat)*std + mean


centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

# X.shape
clusters = np.zeros(m)
distances = np.zeros((m,K))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(K):
        distances[:,i] = np.linalg.norm(X - centers[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(K):
        centers_new[i] = np.mean(X[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new    
# -------------------------------------

    
    

#first time implementation----------------
for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids = np.c_[Centroids,X[rand]]

num_iter=100
Output=defaultdict()
Output={}
for n in range(num_iter):
    #step 2.a
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        
    C=np.argmin(EuclidianDistance,axis=1)+1
    #step 2.b
    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(dim,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
    for k in range(K):
        Y[k+1]=Y[k+1].T
        
        
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
        
    Output=Y
print (C)