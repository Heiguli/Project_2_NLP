
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "part_1"
    / "pubmed_project2_dataset.csv"
)

output_folder = project_root / "part_11"
output_folder.mkdir(parents=True, exist_ok=True)

topic_keywords_path = (
    output_folder / "bertopic_supervised_keywords.csv"
)

document_output_path = (
    output_folder / "bertopic_supervised_documents.csv"
)


# Load dataset
df = pd.read_csv(input_path)

#Get dataframes columns abstract and topic
abstracts = df["abstract"].fillna("").astype(str).tolist()
labels = df["topic"].astype(str).tolist()

#Convert category names to integer labels
#We do this because the model needs numerical versions of labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

#Boilerplate medical terms. Then combine these with default english stopwords
#to create full used stopwords
custom_stop_words = {
    "study", "studies", "results", "result",
    "methods", "method", "conclusion", "conclusions",
    "objective", "objectives", "background",
    "participants", "participant", "included",
    "performed", "conducted", "investigated",
    "examined", "assessed", "analysis", "analyses",
    "significant", "significantly", "association",
    "associations", "relationship", "relationships",
    "compared", "respectively", "baseline",
    "patients", "patient", "individuals",
    "data", "findings"
}

all_stop_words = sorted(
    set(ENGLISH_STOP_WORDS) | custom_stop_words
)

#Vectorizer which we use to create numerical representation
#of words. 
vectorizer = CountVectorizer(
    ngram_range=(1, 3),
    stop_words=all_stop_words,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]*\b"
)

#Bert configuration. We use biomedical model BioLORD
embedding_model = SentenceTransformer(
    "FremyCompany/BioLORD-2023-S"
)

#We use keybert here too for more accurate keywords
representation_model = KeyBERTInspired(
    top_n_words=10
)

#This is the model initialization
#We set up the embedding model it uses (biomedical one)
#Usually bertopic is unsupervised and learns clusters to 
#learn different topics from given inputs. Here we dont do unsupervised, but
#supervised learning and tell the model to not do the usual unsupervised clustering
#Use the vectorizer we created previously. ALso use a tfidf transformer for more accurate
#and better outputs (for example reduce too frequent terms, and increase terms that should be relevant
#by outputting terms which appear not too often, but not too seldom) 
#Use keybert representation model
topic_model = BERTopic(
    embedding_model=embedding_model,

    # Use the supplied labels instead of discovering
    # clusters through dimensionality reduction.
    umap_model=BaseDimensionalityReduction(),
    hdbscan_model=LogisticRegression(max_iter=1000),

    vectorizer_model=vectorizer,
    ctfidf_model=ClassTfidfTransformer(
        reduce_frequent_words=True
    ),
    representation_model=representation_model,

    top_n_words=10,
    verbose=True
)

#Then fit the actual model with the preknown, supervised topics and abstracts

topics, probabilities = topic_model.fit_transform(
    abstracts,
    y=y
)

#Then map bert topic ids to word representations (From ids back to word representation)

topic_mapping = topic_model.topic_mapper_.get_mappings()

topic_to_category = {
    topic_mapping[class_id]: category
    for class_id, category
    in enumerate(label_encoder.classes_)
}

#Get some topic info and print to help understand data more
topic_info = topic_model.get_topic_info()

topic_info["category"] = topic_info["Topic"].map(
    topic_to_category
)

print("\nTopic information:")
print(topic_info[["Topic", "category", "Count", "Representation"]])


#Then we save category(topic) specific keywords. Each topic gets 10 keywords. 
#Then  these 10 keywords will be used to hopefully improvove the returned, extracted
#keywords from each documents abstracts
topic_rows = []

for topic_id, category in topic_to_category.items():
    keywords = topic_model.get_topic(topic_id)

    if not keywords:
        continue

    for rank, (keyword, score) in enumerate(keywords, start=1):
        topic_rows.append({
            "category": category,
            "topic_id": topic_id,
            "rank": rank,
            "keyword": keyword,
            "score": score
        })

topic_keywords_df = pd.DataFrame(topic_rows)

topic_keywords_df.to_csv(
    topic_keywords_path,
    index=False
)



#Save document  topic assignments


df["bertopic_topic_id"] = topics
df["bertopic_category"] = [
    topic_to_category.get(topic_id, "Unknown")
    for topic_id in topics
]

df["bertopic_topic_keywords"] = [
    "; ".join(
        keyword
        for keyword, score in (topic_model.get_topic(topic_id) or [])
    )
    for topic_id in topics
]

df.to_csv(document_output_path, index=False)



#Some printing keywords for each category


for topic_id, category in topic_to_category.items():
    print(f"\n{'=' * 60}")
    print(f"Category: {category} | Topic ID: {topic_id}")
    print("=" * 60)

    for rank, (keyword, score) in enumerate(
        topic_model.get_topic(topic_id) or [],
        start=1
    ):
        print(f"{rank:2}. {keyword:<40} {score:.4f}")


print("\nSaved category keywords to:")
print(topic_keywords_path)

print("\nSaved document assignments to:")
print(document_output_path)
