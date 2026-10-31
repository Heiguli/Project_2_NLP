
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
#requirements pip install scikit-learn

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
    project_root / "part_7" /
    "pubmed_project2_similarities_LDA_task7.csv"
)

df = pd.read_csv(input_path)
statistics_df = pd.read_csv(statistics_path)




#Gives us four abstract groups from each topic, so 4x10 = 40
#We will get 4 different sets of keywords per topic with lda
#based on these abstract groups.
def get_lda_keywords():
    results = []
    for topic, topic_df in df.groupby("topic"):
        abstracts = (
            topic_df["abstract"]
            .dropna()
            .reset_index(drop=True)
            .tolist()
        )
        keyword_groups = []
        for i in range(0, len(abstracts), 25):
            try:
                #LDA model
                lda_model = LatentDirichletAllocation(
                    n_components=1,
                    random_state=42,
                    max_iter=10
                )

                #Vectorizer
                vectorizer = CountVectorizer(
                    stop_words="english",
                    max_df=1.0
                )
                abstract_group = abstracts[i:i + 25]
                document_term_matrix = vectorizer.fit_transform(
                    abstract_group
                )       #builds a vocabulary and changes words to numbers for model

                lda_model.fit(document_term_matrix) #train the model
                feature_names = vectorizer.get_feature_names_out() #get words out (abstract words)
                topic_word_weights = lda_model.components_[0] #get word weights
                top_word_indices = topic_word_weights.argsort()[-10:][::-1] #gets highest weights
                lda_keywords = [
                    feature_names[index]
                    for index in top_word_indices
                ] #Gets the words with highest weights

            # Save this group's keyword list
                keyword_groups.append(lda_keywords)
            except Exception as error:
                print(f"Error in {topic}: {error}")
        results.append({
            "topic": topic,
            "keywords": keyword_groups
        })
    return results

results = get_lda_keywords()

def calculate_lda_consistency(keyword_groups):
    # Convert each keyword list to a set
    keyword_sets = [
        set(keywords)
        for keywords in keyword_groups
    ]

    # Find words shared by ALL four lists
    common_keywords = set.intersection(*keyword_sets)

    # Find all unique words across all four lists
    all_keywords = set.union(*keyword_sets)

    # Calculate consistency as a percentage
    if not all_keywords:
        return 0.0

    consistency = (
        len(common_keywords) / len(all_keywords)
    ) * 100

    return consistency


#We then iterate over results to get each topics single similarity LDA score
def get_similarity_results():
    similarity_results = []
    for item in results:
        topic = item["topic"]
        keyword_groups = item["keywords"]
        similarity = calculate_lda_consistency(keyword_groups)
        similarity_results.append({"topic": topic, "similarity": similarity})
    return similarity_results

similarity_results = get_similarity_results()
similarity_df = pd.DataFrame(similarity_results)
similarity_df.to_csv(output_path_similarity, index=False)

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

    plt.xlabel("LDA topic consistency (v)")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs. LDA topic Consistency")
    plt.legend(title="Algorithm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(project_root /"part_7" / f"pubmed_project2_{metric}_vs_LDA_consistency_task7.png", dpi=300)
    plt.show()