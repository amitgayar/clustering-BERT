import pickle   
import string
from nltk.stem import PorterStemmer

from os import listdir
from os.path import isfile, join
mypath = 'news-analyzer/data/rafale deal/news/'
news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]                                                                                                                                                                      

# -----------------------spacy cleaning

import spacy #load spacy
# nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
stops = sorted(stopwords)

import spacy

nlp = spacy.load('en_core_web_sm')
# Format for accessing doc tuned by 'nlp' :
# 
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)



def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    punct = '( ) : [ ] ? . , | _ ^ - & > < ; " !  / % $ @ * % #'
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
            	if lemma not in punct.split():
            		lemmatized.append(lemma)

                
    return lemmatized



#------------------------------rest compution

bow = [] #........cleaned documents in list of lists
for f in news_text_file:
    
    text_article = pickle.load(open(f, 'rb')) 
    temp = normalize(text_article['content'], lowercase=True, remove_stopwords=True)
    bow.append(temp)
    print ('article no : ',news_text_file.index(f), sorted(temp),'\n')

    

wordSet = sorted(set().union(*bow))
# print (wordSet,'\n',len(wordSet))



# ------------Storing the results
store_file = {}
store_file['wordset'] = wordSet
store_file['clean_text_doclist'] = bow 
pickle.dump(store_file,open('load_words_all_docs_spacy.pkl','wb'))
