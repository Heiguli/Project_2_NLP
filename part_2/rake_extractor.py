import RAKE

#requirements install python -m pip install python-rake   

#RAKE
#Rake returns word with frequence in second index, tuples
#this sorts them by the score
def Sort_Tuple(tup):
    tup.sort(key = lambda x: x[1])
    return tup 
#Uses word freq and co occurence, highest value most important
def rake(text, top_k=10, stopwords_file_path="stop_words_english.txt"):
    """Usage of keywords extraction algorithm RAKE, codes are taken from here
       https://pypi.org/project/python-rake/

    :param str text: row text, without cleaning it from punctuation, stopwords etc.
    :param int top_k: top_k keywords to print
    :Param str stopwords_file_path: path to stopwords list file
    """
    rake_object = RAKE.Rake(stopwords_file_path)
    keywords = Sort_Tuple(rake_object.run(text))
    results = []
    for k in keywords[-top_k:]:
        results.append(k[0])
    return results
