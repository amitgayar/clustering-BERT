import pickle   
import string
from nltk.stem import PorterStemmer

from os import listdir
from os.path import isfile, join
mypath = 'news-analyzer/data/rafale deal/news/'
news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# ---remove punctuations; code from : https://towardsdatascience.com/working-with-text-data-from-quality-to-quantity-1e9d8aa773dd

# ps = PorterStemmer()
# def str_clean(text):
#     punct = '():[]?.,|_^-&><;!"/%'  
#     table = str.maketrans(punct, ' '*len(punct), "0123456789$#'=")
#     cleaned_comment = []
#     # re.findall(r"[\w']+", text)
#     for word in text.split():
#         cleaned_comment.extend(word.translate(table).split())
#         cleaned_comment = [x.lower() for x in cleaned_comment]
#         cleaned_comment = [ps.stem(word) for word in cleaned_comment]
#     return cleaned_comment                                                                                                                                                                       

# ------spacy cleaning

import spacy #load spacy
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
# stopwords = STOP_WORDS
# stops = stopwords.words("english")
stops = sorted(stopwords)




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



#------------rest compution
bow = []
for f in news_text_file:
    
    text_article = pickle.load(open(f, 'rb')) 
    text_article['Text_After_Clean'] = normalize(text_article['content'], lowercase=True, remove_stopwords=True)
    bow.append(text_article['Text_After_Clean'])
    # print (type(text_article['Text_After_Clean']))
    print ('article no : ',news_text_file.index(f), text_article['Text_After_Clean'],'\n')
    # bow.append(str_clean(text_article['content'])) 
print ('\n','type of variable : ',type(bow))
store_file = {}
wordSet = sorted(set().union(*bow))
print (wordSet,'\n',len(wordSet))
store_file['wordset'] = wordSet
store_file['clean_text_doclist'] = bow 

pickle.dump(store_file,open('load_words_all_docs_spacy_second_pruning.pkl','wb'))
