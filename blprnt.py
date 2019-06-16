# ----------------------------------------------------------------------------------------
import pickle   
from os import listdir
from os.path import isfile, join
import spacy #load spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from collections import defaultdict
import matplotlib.cm as cm
from copy import deepcopy

para_text = []
tfidf = []
clusters = []


def spacy_cleansing(comment):
    # Format for accessing doc tuned by 'nlp' :
    # 
    # doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #             token.shape_, token.is_alpha, token.is_stop)
    comment = nlp(comment)
    lemmatized = []
    for token in comment:
        if not token.is_stop and token.is_alpha:
            lemmatized.append(token.lemma_)

                
    return lemmatized

def art_to_para(article,threshold=50):
    #     For fragmentation of article text into paragraphs of word-count >= threshold
    #         Parameters :
    #         -------------
    #         article : str
    #         threshold : int, optional, default 50
	para_list = article.split('\n\n')
    if len(para_list) == 1 and len(para_list[0]) > 2.7*threshold :
        temp = para_list[0].split('. ')
        para_list = [" ".join(temp[:math.floor(len(temp)/2)]), " ".join(temp[math.floor(len(temp)/2):])]
	l = len(para_list)
    for j in para_list:
        para_text[para_list.index(j)] = j
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
            cluster_len = len(clustered_para[-1])
            if cluster_len < threshold and cluster_len > 1:
                clustered_para[-2] = clustered_para[-2] + " " + clustered_para[-1]
                del clustered_para[-1]
			return clustered_para
		else:
			return para_thresholding(i+1,clustered_para)

	return para_thresholding()


def load_and_break(keywords='rafale deal'):
    mypath = 'news-analyzer/data/'+keywords+'/news/'
    news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    usr_input = input("For whole article, Press a \nFor paragraph segmentation, press p\n")
    bow = [] #........cleaned documents in list of lists
    if usr_input == 'a':
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            temp = spacy_cleansing(text_article['content'])
            bow.append(temp)
    #         print ('article no : ',news_text_file.index(f), sorted(temp),'\n')
    else:
        para_count_in_art = []
        para_list = []
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            temp = art_to_para(text_article['content'])
            para_list += text_article['content'].split('\n\n')
            para_count_in_art.append(len(temp))
            for t in temp:
                temp1 = spacy_cleansing(t)
                bow.append(temp1)
    store_file = {}
    store_file['para_count_in_art'] = para_count_in_art if usr_input != 'a' else 0
    store_file['para_list'] = para_list if usr_input != 'a' else 0
    store_file['clean_text_doclist'] = bow
#     if selection == 'a':
#         file = 'spacy_cleansing.pkl'
#     else:
#         file = 'spacy_cleansing_of_paras.pkl'
#     pickle.dump(store_file,open(file,'wb'))
    return store_file


# ---------------------------------- tfidf -------------------------------------------------------------------------------
processed_doc = load_and_break()

# processed_doc = pickle.load(open('spacy_cleansing_of_paras.pkl','rb'))
doc = processed_doc['clean_text_doclist']
wordSet = sorted(set().union(*doc))

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
    	tfDict[word] = round((count+1)/doc_count,4)
#     print(str(tfDict),'\n')
    return tfDict

tfdoc = []
for i in range(len(doc)):
    tfdoc.append(computeTF(doc_set_all[i], doc[i]))


def computeIDF(doc_set_all,wordSet):
    import math
    idfDict = dict.fromkeys(wordSet, 0)
    N = len(doc_set_all)
    for word in wordSet:
        for i in range(N):
        	if word in doc_set_all[i].keys():
        		idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / val)
                        
    return idfDict        


idfs = computeIDF(doc_set_all,wordSet)

def computeTFIDF(tfdoc, idfs,threshold=0.01):
    tfidf = {}
    new_wordset = [] #for thresholding
    for word, val in tfdoc.items():
        ti = round(val*idfs[word],4)
        if ti > threshold:
            tfidf[word] = ti #------rounded for error handeling of data(float64)
            new_wordset.append(word)
#     print(str(tfidf),'\n')
#     print(str(new_wordset),'\n')
    return tfidf,new_wordset

def main_tfidf(threshold=0.01):
    tfidf = []
    new_wordSet = []
    for t in tfdoc:
        temp = computeTFIDF(t, idfs,threshold)
        new_wordSet += temp[1]
        tfidf.append(temp[0])
    wordSet = sorted(set().union(new_wordSet))
    # print(len(wordSet))

    tfidf_wordSet = [dict.fromkeys(wordSet, 0)]*len(doc)



    i=0 
    tfidf_final_struc = [] 
    for wrdF in tfidf_wordSet: 
        temp = tfidf[i] 
        # print(temp, wrdF) 
        tfidf_final_struc.append(dict(wrdF, **temp)) 
        i += 1


    df = pd.DataFrame(tfidf_final_struc)

    # df.to_csv('tfidf_of_paras.csv') 
    return df, tfidf

tfidf_final = main_tfidf()
#------------------------------------------- k clustering 2 --------------------------------------------------------
def main_clustering():
    # https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python
    # df=pd.read_csv('tfidf_of_paras.csv')
    df = tfidf_final[0]
    X = df.iloc[:,1:].values
    X = np.nan_to_num(X)
    K=5
    m, feat = X.shape
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    centers = np.random.randn(K,feat)*std + mean
    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers
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
    # centers_new 
    clusters
    return clusters.tolist()

clusters = main_clustering()

for i in clusters:
    print (i, '\n', processed_doc['para_list'][i],'\n', tfidf_final[1][i], '\n')

# ---------------------------------------rough work---------------------------------------------------------------------

"""

# paragraph :
def art_to_para(article):
	para_list = article.split('\n\n')
	l = len(para_list)
	def para_thresholding(i=0, clustered_para=[]):
		temp = para_list[i]
        
        
        def mar(temp):
            import math
            if len(temp.split())>140:
                temp = temp.split('. ')
                temp1 = mar(' '.join(temp[:math.floor(len(temp)/2)]))
                temp2 = mar(' '.join(temp[math.ceil(len(temp)/2):]))
                return temp1, temp2
            return temp
        mar(temp)
        
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
"""