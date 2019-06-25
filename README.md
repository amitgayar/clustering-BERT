# clustering-BERT
tfidf + clustering...
1. "news-analyzer/core.py" loads the various articles based on input keywords
2. "all_in_one.ipynb" does the clustering of the text extracted while accepting tuning parameters like min_tfidf value, para-segmentation.
3. Clustering is Divisive with successive bipartitions of the text chunk groups to obtain the desired level of clustering.
4. 