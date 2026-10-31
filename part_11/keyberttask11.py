
import pandas as pd
from pathlib import Path
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "part_1"
    / "pubmed_project2_dataset.csv"
)

output_folder = project_root / "part_11"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "keybert_top20_results.csv"


# Load the dataset
df = pd.read_csv(input_path)
abstracts = df["abstract"].fillna("").astype(str).tolist()

#Load the transformer embedding model, we use a biomedical embedding model because our 
#documents are medically related
embedding_model = SentenceTransformer(
    "FremyCompany/BioLORD-2023-S"
)

#Common academic boilerplate, not usually useful as keywords so we 
#remove them from possible keywords. Could add more, this is enough
custom_stop_words = [
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
]

#Combine the default english stop words and the boilerplate keywords  
all_stop_words = sorted(
    set(ENGLISH_STOP_WORDS) | set(custom_stop_words)
)

#create the vectorizer which gives us numerical representation
#of words. Trigram words allowed, stop words is the combined stopwords,
#token pattern means skipping numbers but still trying to allow medical terms etc..
vectorizer = CountVectorizer(
    ngram_range=(1, 3),
    stop_words=all_stop_words,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]*\b"
)

#Initialize KeyBERT, the embedding model
kw_model = KeyBERT(model=embedding_model)

#Extract the top 20 keyword/keyphrase candidates per abstract
all_keywords = kw_model.extract_keywords(
    abstracts,
    vectorizer=vectorizer,
    top_n=20,
    use_mmr=True,
    diversity=0.5
)

#Save predictions in a new column in the dataframe
df["keybert_keywords"] = [
    "; ".join(keyword for keyword, score in doc_keywords)
    for doc_keywords in all_keywords
]

#Optionally retain similarity scores for inspection
df["keybert_scored"] = [
    "; ".join(f"{keyword} ({score:.3f})"
              for keyword, score in doc_keywords)
    for doc_keywords in all_keywords
]

#Save the results without modifying the original dataset

df.to_csv(output_path, index=False)

#Display a few examples (Will remove before final implementation)
for i in range(min(3, len(df))):
    print(f"\n--- Abstract {i + 1} ---")
    print(df.loc[i, "abstract"][:400], "...")
    print("\nTop KeyBERT keywords:")
    for rank, (keyword, score) in enumerate(all_keywords[i], start=1):
        print(f"{rank:2}. {keyword:<40} {score:.3f}")

print(f"\nSaved results to: {output_path}")
 

