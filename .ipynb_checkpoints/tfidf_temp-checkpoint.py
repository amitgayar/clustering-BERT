import pickle 

processed_doc = pickle.load(open('load_words_all_docs_spacy.pkl','rb'))
wordSet = processed_doc['wordset']
doc = processed_doc['clean_text_doclist']

doc_set_all = []
for docc in doc:
	doc_set = sorted(set().union(docc))
	doc_dict = dict.fromkeys(doc_set, 0)
	for word in docc:
		doc_dict[word] += 1 
	doc_set_all.append(doc_dict.copy())

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
        tfidf[word] = val*idfs[word]
    return tfidf

tfidf = []
for t in tfdoc:
    tfidf.append(computeTFIDF(t, idfs))

import pandas as pd
df = pd.DataFrame([i for i in tfidf])

