import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def get_idf_keywords():
    #At start missed to add the tf-idf extraction method so doing it now at task 9
    project_root = Path(__file__).resolve().parent.parent

    input_path = project_root / "part_2" / "pubmed_project2_with_keywords.csv"

    df = pd.read_csv(input_path)

    # Create a column to store the TF-IDF keywords
    df["tfidf"] = ""

    # Process each topic separately
    for topic, topic_df in df.groupby("topic"):

        abstracts = topic_df["abstract"].fillna("").tolist()

        # Fit TF-IDF only on documents belonging to this topic
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        #get words from all topic abstracts
        words = vectorizer.get_feature_names_out()

        # Extract the top 10 words from each abstract
        for row_position, original_index in enumerate(topic_df.index):
            #get scores for abstract that is in row_position (abs 0, abs 1 etc..)
            scores = tfidf_matrix[row_position].toarray()[0]
            #get the indexes of the top 10 scores in desc order
            top_indices = scores.argsort()[::-1][:10]
            #then because the scores were at the same indexes as the words, 
            #we use those previously gotten indexes to get the words with
            #the highest score in desc order
            keywords = [
                words[i]
                for i in top_indices
                if scores[i] > 0
            ]
            #save this in dataframe df in correct position, correct cell
            df.at[original_index, "tfidf"] = json.dumps(
                keywords, ensure_ascii=False
            )

    #Save the dataset with the new TF-IDF column
    df.to_csv(input_path, index=False)
    print(f"Saved tf-idf keywords!")