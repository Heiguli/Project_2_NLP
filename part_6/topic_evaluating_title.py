
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#requirements pip install -U sentence-transformers

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root / "part_2" /
    "pubmed_project2_with_keywords.csv"
)

statistics_path = (
    project_root / "part_3" /
    "pubmed_project2_with_keywords_with_statistics.csv"
)

output_path_similarity = (
    project_root / "part_6" /
    "pubmed_project2_similarities_title_task6.csv"
)

df = pd.read_csv(input_path)
statistics_df = pd.read_csv(statistics_path)

#Using SBERT since its better for semantic similarity  projects
#and not using the original BERT
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#This gives topic consistency SBERT vectors
#results contains topic and then the topics 100 difrerent SBERT vectors
def get_embedding_vectors():
    results = []
    for topic, topic_df in df.groupby("topic"):
        bert_embeddings = []
        for _, row in topic_df.iterrows():
            try:
                if pd.isna(row["title"]):
                    continue

                title = str(row["title"])
                #We normalize the vectors because we want to focus on the vector directions and not 
                #the magnitude of the vectors lenght and because of this, each inner product is a cosine similarity
                bert_embedding = model.encode(title, normalize_embeddings=True)
                bert_embeddings.append(bert_embedding)
            except Exception as error:
                print(f"Error in {topic}: {error}")
        if bert_embeddings:
            results.append({
                "topic": topic,
                "bert_embeddings": bert_embeddings
            })
    return results

results = get_embedding_vectors() 

def calculate_topic_consistency(embeddings):
    embeddings = np.array(embeddings)
    # Calculate all pairwise inner products by matrix multiplication
    similarity_matrix = embeddings @ embeddings.T

    # Extract only unique pairs, excluding self-comparisons
    upper_triangle = similarity_matrix[
        np.triu_indices(len(embeddings), k=1)
    ]

    # Average pairwise inner products
    return float(np.mean(upper_triangle))


#get the topic similarities into a array
def get_similarity_results():
    similarity_results = []
    for item in results:
        topic = item["topic"]
        sbert_embeddings = item["bert_embeddings"]
        similarity = calculate_topic_consistency(sbert_embeddings)
        similarity_results.append({"topic": topic, "similarity": similarity})
    return similarity_results

similarity_results = get_similarity_results()
similarity_df = pd.DataFrame(similarity_results)
similarity_df.to_csv(output_path_similarity, index=False)
#combine statistic and similarity dataframes
combined_df = statistics_df.merge(
    similarity_df,
    on="topic",
    how="inner"
)


metrics = ["precision", "recall", "f1"]

# Create one separate plot for each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))

    # Plot each extraction algorithm separately
    for extractor in combined_df["extractor"].unique():
        extractor_df = combined_df[
            combined_df["extractor"] == extractor
        ].sort_values("similarity") #Sort the rows from lowest similarity to highest so when plotting, 
                                    #similarity grows on x-axis so when moving from left to right

        plt.plot(
            extractor_df["similarity"],
            extractor_df[metric],
            marker="o", #Draws a mark on each topics data point
            label=extractor
        )

    plt.xlabel("Topic consistency based on titles (v)")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs. Title-Based Topic Consistency")
    plt.legend(title="Algorithm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(project_root / "part_6" / f"pubmed_project2_{metric}_vs_Title_consistency_task6.png", dpi=300)
    plt.show()