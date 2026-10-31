from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
import string
import nltk 

#  Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
nltk.download("punkt")
nltk.download("punkt_tab")
def filter_keywords_by_title_semantic(title, keywords, top_n=10):

    if not isinstance(title, str) or not title.strip():
        return keywords[:top_n]

    if not keywords:
        return []

    #Split the title into tokens and remove ".","," etc..  tokens
    title_tokens = [
        token for token in word_tokenize(title)
        if token not in string.punctuation
    ]

    if not title_tokens:
        return keywords[:top_n]


    #Remove empty or keywords that are not strings
    keyword_texts = [
        keyword for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    ]

    if not keyword_texts:
        return []

    #use encode to get the numerical values of keywords and title
    keyword_embeddings = model.encode(
        keyword_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    title_embeddings = model.encode(
        title_tokens,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    #Compare every candidate with every title token
    similarities = util.cos_sim(
        keyword_embeddings, title_embeddings
    )

    #maximum similarity for each candidate
    max_scores = similarities.max(dim=1).values

    #Sort candidates from highest to lowest score
    ranked_indices = max_scores.argsort(descending=True)

    return [
        keyword_texts[i]
        for i in ranked_indices[:top_n].tolist()
    ]

