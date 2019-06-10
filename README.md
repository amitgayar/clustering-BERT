# clustering-BERT
tfidf k-mean clustering...
1. "news-analyzer/core.py" loads the various articles based on input keywords
2. "load_words_all_docs.py" cleans the content of all articles saved in "news-analyzer/data" directory
    and saves in "load_words_all_docs_spacy.pkl"
3. "tfidf.py" computes tfidf of the cleaned content and stores in "sample_tfidf.csv"
4. K-means clustering to be done ...

note: "temp" directory is for old/obsolete results
