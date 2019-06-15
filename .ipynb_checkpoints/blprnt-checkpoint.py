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