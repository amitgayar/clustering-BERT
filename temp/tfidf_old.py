import pickle   
# import string
# from nltk.stem import PorterStemmer

# from os import listdir
# from os.path import isfile, join
# mypath = '../news-analyzer/data/rafale deal/news/'
# news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# # ---remove punctuations; code from : https://towardsdatascience.com/working-with-text-data-from-quality-to-quantity-1e9d8aa773dd
# ps = PorterStemmer()
# def str_clean(text):
#     punct = '():[]?.,|_^-&><;!"/%'  
#     table = str.maketrans(punct, ' '*len(punct), "0123456789$#'=")
#     cleaned_comment = []
#     for word in text.split():
#         cleaned_comment.extend(word.translate(table).split())
#         cleaned_comment = [ps.stem(word) for word in cleaned_comment]
#     return cleaned_comment                                                                                                                                                                       

# # ---load files
# doc = []
# for f in news_text_file:
#     text_article = pickle.load(open(f, 'rb')) 
#     doc.append(str_clean(text_article['content']))
# text_articleA = pickle.load(open(news_text_file[0], 'rb'))
# text_articleB = pickle.load(open(news_text_file[1], 'rb'))



# docA = str_clean(text_articleA['content'])
# docB = str_clean(text_articleB['content'])

# print (doc[0:4])

# docA = docA.split(" ")
# docB = docB.split(" ")

# wordSet = set(doc[0]).union(set(doc[1]))
# wordSet = set().union(*doc)
# print (sorted(wordSet1))
processed_doc = pickle.load(open('load_words_all_docs_spacy.pkl','rb'))
wordSet = processed_doc['wordset']
doc = processed_doc['clean_text_doclist']

wordDict = dict.fromkeys(wordSet, 0)
wordDict = [wordDict]*len(doc)
# print (wordDict)
# wordDictA = dict.fromkeys(wordSet, 0)
# wordDictB = dict.fromkeys(wordSet, 0)

for i in range(88):
    for word in doc[i]:    	
        wordDict[i][word]+=1 #-------to be checked
        





# -------------------------------temp



if wordDict[temp].values == 1:
    print (wordDict[temp].values())

for k,v in wordDict[temp].items(): 
    if v>50: 
        print (v,k)

for count in range(20):
    i = count
    for word in doc[i]:
        print(wordDict[i][word],word)
   
     

# --------------------------------------


        # for word in doc[temp]: 
        #     print(wordDict[temp][word],word)
        
# for word in doc[1]:
#     wordDictB[word]+=1

# import pandas as pd
# pd.DataFrame([wordDictA, wordDictB])
# pd.DataFrame([w for w in wordDict])


def computeTF(wordDict, doc):
    tfDict = {}
    tfDict = dict.fromkeys(doc, 0)
    docCount = len(doc)
    for word in sorted(doc):

    for word, count in wordDict.items():
        tfDict[word] = count/(float(docCount)+1) # =============check for +1
    return tfDict
tfdoc = []
for i in range(len(doc)):
    tfdoc.append(computeTF(wordDict[i], doc[i]))
# tfdocA = computeTF(wordDict[0], doc[0])
# tfdocB = computeTF(wordDict[1], doc[1])
# print (tfdocB,"\n",tfdocA)

import pandas as pd
# df = pd.DataFrame([tfdocA, tfdocB])
df = pd.DataFrame([i for i in tfdoc])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
                
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1
             
    for word, val in idfDict.items():
        print 
        idfDict[word] = math.log10(N / float(val))
                        
    return idfDict        

idfs = computeIDF([i for i in wordDict])

def computeTFIDF(tfdoc, idfs):
    tfidf = {}
    for word, val in tfdoc.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidf = []
for t in tfdoc:
    tfidf.append(computeTFIDF(t, idfs))

# tfidfdocA = computeTFIDF(tfdocA, idfs)
# tfidfdocB = computeTFIDF(tfdocB, idfs)
# print (tfidfdocB,"\n",tfidfdocA)


df = pd.DataFrame([i for i in tfidf])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print("\n",df)

# from pandas import ExcelWriter
# export_excel = df.to_excel (r'sample_tfidf.xlsx', index = None, header=True)

