
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter, OrderedDict

#requirements pip install pandas

#Preprocessing
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

#Helper preprocessing function, makes text lowercase,
#removes digits, removes punctuation, splits into words and removes
#stopwords and lemmatizes the text
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]

    words = [lemmatizer.lemmatize(word) for word in words]

    return words
#luhn kws extractor. Extracts kws based on the frequence of words. 
#Skips words that occur too often
def luhn(text, num_to_take=100):
    text = preprocess_text(text)
    """Implementation of Luhn algorithm described here
       https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf

    :param str text: already preprocessed text (without digits, punctuations, stopwords, lemmatized etc.)
    :param int num_to_take: how many extracted words to return
    """
    frequency = OrderedDict({})
    N, i = len(text), 0

    for word in text:
        count = frequency.get(word , 0)
        frequency[word] = count + 1
    sorted_dict = OrderedDict({})
    sorted_keys = sorted(frequency, key=frequency.get, reverse=True)

    for w in sorted_keys:
        sorted_dict[w] = frequency[w]
    res_dict = Counter(sorted_dict.values())
    max_rank = None
    freq = 1
    for k, v in sorted_dict.items():
        max_rank = len(frequency.keys())
        if k not in frequency.keys():
            print("skipping ", k)
            continue
        if res_dict[freq] >= 10:
            for k1, v1 in list(sorted_dict.items()):
                if v1 == freq:
                    frequency.pop(k1, "no such value")
        else:
            break
        freq+=1
    
    luhn_results = sorted(frequency, key=frequency.get, reverse=True)[:num_to_take]
    return luhn_results

