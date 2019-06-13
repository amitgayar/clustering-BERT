import pickle 

processed_doc = pickle.load(open('load_words_all_docs_spacy.pkl','rb'))
wordSet = processed_doc['wordSet']
doc = processed_doc['clean_text_doclist']

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

def computeTFIDF(tfdoc, idfs):
    tfidf = {}
    for word, val in tfdoc.items():
        tfidf[word] = round(val*idfs[word],4) #------rounded for error handeling of data(float64)
    return tfidf

tfidf = []
for t in tfdoc:
    tfidf.append(computeTFIDF(t, idfs))

tfidf_wordSet = [dict.fromkeys(wordSet, 0)]*len(doc)

# these two for loops aren't working the way they are supposed to:
# 1)
#         for i in range(len(doc)):
# 	          tfidf_wordSet[i].update(**tfidf[i])
# 2)
#         for i in range(len(doc)):
# 	          temp = tfidf[i]
# 	          tfidf_wordSet[i].update(**temp)
# 
# ------------ the solution ------------------
#  
#         for i in range(len(doc)):
# 	          temp = tfidf[i]
# 	          tfidf_wordSet[i] = dict(tfidf_wordSet[i],**temp)




i=0 
tfidf_final_struc = [] 
for wrdF in tfidf_wordSet: 
    temp = tfidf[i] 
    # print(temp, wrdF) 
    tfidf_final_struc.append(dict(wrdF, **temp)) 
    i += 1


import pandas as pd
# from pandas import ExcelWriter
df = pd.DataFrame(tfidf_final_struc)

df.to_csv('tfidf.csv') 
