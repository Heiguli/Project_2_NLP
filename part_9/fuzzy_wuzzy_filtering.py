from fuzzywuzzy import fuzz

#Fuzzy wuzzy filtering
def filter_keywords_by_title(title, keywords, top_n=10):
    if not isinstance(title, str) or not title.strip():
        keywords = []
        return keywords

    ranked_keywords = []
    ranked_keywords_only = []
    for keyword in keywords:
        score = fuzz.token_set_ratio(keyword, title)
        ranked_keywords.append((keyword, score))

    # Sort by score, highest first
    ranked_keywords.sort(key=lambda item: item[1], reverse=True)
    for keyword, score in ranked_keywords:
        ranked_keywords_only.append(keyword)
    # Return only the keyword strings
    return ranked_keywords_only[:top_n]