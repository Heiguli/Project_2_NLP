
import yake

#requirements pip install yake

#Yake preprocessing, returns 10 keywords
#Multiple local statistical features, lowest score most important
#for example. word relations to other terms, frequence in occurence and
#where it appears
def YAKE(text, top_k=10):
    """YAKE keywords extracton algorithm taken from here
       https://pypi.org/project/yake/

    """
    #Used the unprocessed abstract since yake does its own preprocessing
    kw_extractor = yake.KeywordExtractor(n=3, top=top_k)
    keywords = kw_extractor.extract_keywords(text)
    yake_results = []
    for kw in keywords:
        yake_results.append(kw[0])

    return yake_results


