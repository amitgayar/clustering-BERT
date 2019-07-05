
import pickle   
from os import listdir
from os.path import isfile, join
import spacy #load spacy
nlp = spacy.load('en_core_web_sm')

def spacy_cleansing(doc):
    # for token in doc:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #             token.shape_, token.is_alpha, token.is_stop)
    doc = nlp(doc)
    lemmatized = []
    for token in doc:
        if not token.is_stop and token.pos_ not in ['SYM']:
            lemmatized.append(token.lemma_)            
    return lemmatized


def art_to_para(article,threshold=50):
    #     For fragmentation of article text into paragraphs of word-count >= threshold
    #         Parameters :
    #         -------------
    #         article : str
    #         threshold : int, optional, default 50
    para_list = article.split('\n\n')
    if len(para_list) == 1 and len(para_list[0]) > 3.2*threshold :
        temp = para_list[0].split('. ')
        para_list = [". ".join(temp[:math.floor(len(temp)/2)]), ". ".join(temp[math.floor(len(temp)/2):])]
    l = len(para_list)
    def para_thresholding(i=0, clustered_para=[]):
        temp = para_list[i]
        while len(temp.split(" ")) < threshold and i < l - 1:
            temp = temp + " " + para_list[i+1]
            i+=1
        clustered_para.append(temp) 
        if i == l-1:
            cluster_len = len(clustered_para[-1])
            if cluster_len < threshold and cluster_len > 1:
                clustered_para[-2] = clustered_para[-2] + " " + clustered_para[-1]
                del clustered_para[-1]
            return clustered_para
        else:
            return para_thresholding(i+1,clustered_para)
    return para_thresholding()

def load_and_break(keywords, filesave=False):
    mypath = 'news-analyzer/data/'+keywords+'/news/'
    news_text_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    usr_input = input("For whole article, Press a \nFor paragraph segmentation, press p\n")
    bow = [] #........cleaned documents in list of lists
    text = []
    if usr_input == 'a':
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            text.append(text_article['content'])
            temp = spacy_cleansing(text_article['content'])
            bow.append(temp)
    #         print ('article no : ',news_text_file.index(f), sorted(temp),'\n')
    else:
        para_count_in_art = []
        for f in news_text_file:    
            text_article = pickle.load(open(f, 'rb'))
            temp = art_to_para(text_article['content'])
            text += temp
            para_count_in_art.append(len(temp))
            for t in temp:
                temp1 = spacy_cleansing(t)
                bow.append(temp1)
    store_file = {}
    store_file['para_count_in_art'] = para_count_in_art if usr_input != 'a' else 0
    store_file['text'] = text if usr_input != 'a' else 0
    store_file['clean_text_doclist'] = bow
    if filesave:    
        if usr_input == 'a':
            file = 'spacy_cleansing.pkl'
        else:
            file = 'spacy_cleansing_of_paras.pkl'
        pickle.dump(store_file,open(file,'wb'))
    return store_file


if __name__ == '__main__':
    processed_doc = load_and_break('rafale deal')