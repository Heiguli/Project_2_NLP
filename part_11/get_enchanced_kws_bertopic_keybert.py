
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import json
#Load the data so we can access the keybert kws and bertopic kws
project_root = Path(__file__).resolve().parent.parent
part_11_folder = project_root / "part_11"

keybert_path = part_11_folder / "keybert_top20_results.csv"
topic_keywords_path = part_11_folder / "bertopic_supervised_keywords.csv"

#Load KeyBERT and BERTopic results
df = pd.read_csv(keybert_path)
topic_df = pd.read_csv(topic_keywords_path)

#Deal with missing values
df["abstract"] = df["abstract"].fillna("").astype(str)
df["topic"] = df["topic"].astype(str)
df["keybert_keywords"] = (
    df["keybert_keywords"].fillna("").astype(str)
)

topic_df["category"] = topic_df["category"].astype(str)
topic_df["keyword"] = topic_df["keyword"].fillna("").astype(str)

#Get the topic keywords for each topic. So each topic gets 10 keywords.
#We will use these 10 keywords to enchance the keybert kws

#contains topic keywords
topic_terms = {}

for category, group in topic_df.groupby("category"):
    group = group.sort_values("rank")
    topic_terms[category] = group["keyword"].tolist()

missing_categories = set(df["topic"]) - set(topic_terms)

if missing_categories:
    raise ValueError(
        f"No topic keywords found for: {missing_categories}"
    )



#keybert keywords that we got from keybert extracting
#we do little preprocessing for the keywords here
candidate_lists = []

for value in df["keybert_keywords"]:
    candidates = [
        phrase.strip()
        for phrase in value.split(";")
        if phrase.strip()
    ]

    #Remove duplicate phrases while preserving order
    candidates = list(dict.fromkeys(candidates))
    candidate_lists.append(candidates)


# biomedical embedding model
model = SentenceTransformer("FremyCompany/BioLORD-2023-S")



#Encode abstracts. And then encode candidates and topic terms together
#We can access each phrase, topic phrase or keybert phrase by phrase indexing
all_phrases = set()

for candidates in candidate_lists:
    all_phrases.update(candidates)

for terms in topic_terms.values():
    all_phrases.update(terms)

all_phrases = sorted(all_phrases)

#Get the abstract numerical vector representations
print("Encoding abstracts...")
abstract_embeddings = model.encode(
    df["abstract"].tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

#Get the phrases numerical vector representations
print("Encoding candidate and topic phrases...")
phrase_embeddings = model.encode(
    all_phrases,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

#Get phrase, and the phrases index in a dictionary. This  makes it 
#possible for later on to get phrase index to get embedding data
phrase_index = {
    phrase: i for i, phrase in enumerate(all_phrases)
}



#Define score normalization so we can use this to force
#values to be in the range  0-1. This is needed because some 
#values might be outside the range of each other, then we normalize,
#they are both between  0-1
def minmax_normalize(scores):
    scores = [float(s) for s in scores]

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum - minimum < 1e-12:
        return [0.5] * len(scores)

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]



#Rerank candidates using topic information
alpha = 0.70 #how much weight will document sim score, and topic sim score get
enhanced_keywords = []
enhanced_scored = []

#Go through each abstracts keywords in a for loop
for row_index, candidates in enumerate(candidate_lists):
    if not candidates:
        enhanced_keywords.append("")
        enhanced_scored.append("")
        continue

    #get the topic of current abstracts row
    category = df.iloc[row_index]["topic"]

    #get the keybert keywords indexes from the previously created
    #phrase_index list, so we can get embedding vector values
    candidate_ids = [
        phrase_index[phrase] for phrase in candidates
    ]
    #embeddings vector values of this abstract's KeyBERT keywords
    candidate_embeddings = phrase_embeddings[candidate_ids]

    #similarity between abstract and keybert keywords. calculated with cosine similarity
    document_scores = util.cos_sim(
        abstract_embeddings[row_index].unsqueeze(0),
        candidate_embeddings
    )[0].tolist()

    #get topic terms for current documents (abstracts) topic
    category_terms = topic_terms[category]
    #get the phrase ids for embedding vector values
    category_ids = [
        phrase_index[term]
        for term in category_terms
        if term in phrase_index
    ]
    #get embedding vector values for topic keywords
    category_embeddings = phrase_embeddings[category_ids]

    #similarity between keybert keyword candidates
    #and for the current abstracts topic keywords
    candidate_topic_matrix = util.cos_sim(
        candidate_embeddings,
        category_embeddings
    )

    #Take each candidate's three strongest topic-term
    #similarities to reduce reliance on a single match and 
    #then take the mean value of those three similarities and make that
    #the topic similarity
    topic_scores = []

    for candidate_similarities in candidate_topic_matrix:
        similarities = sorted(
            candidate_similarities.tolist(),
            reverse=True
        )

        top_matches = similarities[:min(3, len(similarities))]
        topic_scores.append(
            sum(top_matches) / len(top_matches)
            if top_matches else 0.0
        )

    #Normalize both components within this abstract
    document_norm = minmax_normalize(document_scores)
    topic_norm = minmax_normalize(topic_scores)

    #Combine the document-specific and topical evidence
    #with a weighted formula. Document relevance is much higher
    #(0.7) when compared to topic similarity relevance (0.3)
    #This is the final "score" for the keybert keywords
    combined_scores = [
        alpha * doc_score + (1 - alpha) * topic_score
        for doc_score, topic_score
        in zip(document_norm, topic_norm)
    ]
    #Sort from highest combined score to lowest
    ranked = sorted(
        zip(candidates, combined_scores, document_scores, topic_scores),
        key=lambda item: item[1],
        reverse=True
    )
    ranked = ranked[:10]
    enhanced_keywords.append(
        "; ".join(keyword for keyword, _, _, _ in ranked)
    )

    enhanced_scored.append(
        "; ".join(
            f"{keyword} (combined={combined:.3f}, "
            f"document={doc_score:.3f}, topic={topic_score:.3f})"
            for keyword, combined, doc_score, topic_score in ranked
        )
    )

    if (row_index + 1) % 100 == 0:
        print(f"Processed {row_index + 1}/{len(df)} abstracts.")


#Save enhanced  keyword results

df["topic_enhanced_keywords"] = enhanced_keywords
df["topic_enhanced_scored"] = enhanced_scored

output_path = (
    part_11_folder / "topic_enhanced_keybert_results.csv"
)

df.to_csv(output_path, index=False)

print("\nFinished.")
print("Saved results to:")
print(output_path)

#Get all extraction data in same csv for final comparison between this extraction method and task2 methods
part_2_path = (
    project_root
    / "part_2"
    / "pubmed_project2_with_keywords.csv"
)

output_path_compare = (
    part_11_folder
    / "pubmed_project2_with_all_keyword_extraction_methods_and_with_statistics.csv"
)

df_input = pd.read_csv(part_2_path)

enhanced_keywords = [keyword.split(";") for keyword in enhanced_keywords]
#Convert each semicolon-separated keyword string into a JSON list.
df_input["keybert_bertopic_keywords"] = [
    json.dumps(keywords) for keywords in enhanced_keywords
]

df_input.to_csv(output_path_compare, index=False)

print("Saved combined dataset to:")
print(output_path_compare)