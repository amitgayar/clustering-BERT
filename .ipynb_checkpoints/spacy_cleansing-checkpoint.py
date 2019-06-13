import pickle   
from os import listdir
from os.path import isfile, join
import spacy #load spacy
nlp = spacy.load('en_core_web_sm')

def normalize(comment):
# -----------------------spacy cleaning

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


#------------------------------rest compution

mypath = 'news-analyzer/data/rafale deal/news/'
news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]                                                                                                                                                                      

bow = [] #........cleaned documents in list of lists
for f in news_text_file:
    
    text_article = pickle.load(open(f, 'rb')) 
    temp = normalize(text_article['content'])
    bow.append(temp)
    print ('article no : ',news_text_file.index(f), sorted(temp),'\n')

    

wordSet = sorted(set().union(*bow))
print (wordSet,'\n',len(wordSet))



# ------------Storing the results
store_file = {}
store_file['wordSet'] = wordSet
store_file['clean_text_doclist'] = bow 
pickle.dump(store_file,open('spacy_cleansing.pkl','wb'))