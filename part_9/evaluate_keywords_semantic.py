import pandas as pd
import json
from pathlib import Path
from part_3.keyword_statistics_extractor import get_keyword_extraction_statistics

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root / "part_9" /
    "pubmed_project2_with_keywords_title_semantic.csv"
)

output_path = (
    project_root / "part_9" /
    "pubmed_project2_with_keywords_title_semantic_with_statistics.csv"
)

df = pd.read_csv(input_path)


extractor_keys = [
    "rake_title_semantic",
    "luhn_title_semantic",
    "yake_title_semantic",
    "multipartiterank_title_semantic",
    "positionrank_title_semantic",
    "tpr_title_semantic",
    "singlerank_title_semantic",
    "textrank_title_semantic",
    "topicrank_title_semantic",
    "tfidf_title_semantic"
    ]

results = []

#This gives us information of what precision, recall and f1 each extraction has 
# Data is extraction name, topic name, and then mean precision, recal and f1 across all documents of that topic
for extractor_key in extractor_keys:

    print(f"Running {extractor_key}...")

     #10 rows, each row 

    for topic, topic_df in df.groupby("topic"):
        total_precision = []
        total_recall = []
        total_f1 = []
        for _, row in topic_df.iterrows():
            try:
                if pd.isna(row["keywords"]):
                    reference_keywords = []
                else:
                    reference_keywords = str(row["keywords"]).split(";")
                if pd.isna(row[extractor_key]):
                    extractor_keywords = []
                else:
                    extractor_keywords = json.loads(row[extractor_key])
                precision, recall, f1 = get_keyword_extraction_statistics(reference_keywords, extractor_keywords)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(f1)
            except Exception as error:
                print(f"Error in {extractor_key}, row {topic}: {error}")
        if total_precision:
            results.append({
                "extractor": extractor_key,
                "topic": topic,
                "precision": sum(total_precision) / len(total_precision),
                "recall": sum(total_recall) / len(total_recall),
                "f1": sum(total_f1) / len(total_f1)
            })

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Finished! Results saved to: {output_path}")
print(results_df)


print(f"All extractors finished. Saved dataset to: {output_path}")