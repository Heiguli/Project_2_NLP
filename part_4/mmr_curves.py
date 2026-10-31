
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent

input_path = project_root / "part_2" / "pubmed_project2_with_keywords.csv"
output_path = project_root / "part_4" / "pubmed_project2_mrr_stats.csv"
plot_path = project_root / "part_4" / "pubmed_project2_mrr_curves.png"

df = pd.read_csv(input_path)


extractor_keys = [
    'rake','luhn','yake','multipartiterank','positionrank','tpr','singlerank','textrank','topicrank', 'tfidf'
    ]

def get_keyword_extraction_mmr(reference_keywords, extractor_keywords):

    reference_keywords = {
        key.strip().lower()
        for key in reference_keywords
        if key.strip()
    }

    extractor_keywords = [
        key.strip().lower()
        for key in extractor_keywords
        if key.strip()
    ]
    for rank, keyword in enumerate(extractor_keywords, start=1):
        if keyword in reference_keywords:
            return rank


    return None

results = []

#This gives us information of what precision, recall and f1 each extraction has 
# Data is extraction name, topic name, and then mean precision, recal and f1 across all documents of that topic
for extractor_key in extractor_keys:

    print(f"Running {extractor_key}...")

    first_correct_ranks = []
    for _, row in df.iterrows():
        try:
            if pd.isna(row["keywords"]):
                reference_keywords = []
            else:
                reference_keywords = str(row["keywords"]).split(";")
            if pd.isna(row[extractor_key]):
                extractor_keywords = []
            else:
                extractor_keywords = json.loads(row[extractor_key])
            rank = get_keyword_extraction_mmr(reference_keywords, extractor_keywords)
            first_correct_ranks.append(rank)
        except Exception as error:
            print(
                f"Error in {extractor_key}, "
                f"topic {row['topic']}: {error}"
            )
    if first_correct_ranks:
        for k in range(1, 11):
            reciprocal_ranks = [
                1 / rank if rank is not None and rank <= k else 0
                for rank in first_correct_ranks
            ]

            results.append({
                "extractor": extractor_key,
                "k": k,
                "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
                "documents_evaluated": len(first_correct_ranks)
            })

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

plt.figure(figsize=(10, 6))

for extractor_key in extractor_keys:
    algorithm_results = results_df[
        results_df["extractor"] == extractor_key
    ]

    if not algorithm_results.empty:
        plt.plot(
            algorithm_results["k"],
            algorithm_results["mrr"],
            marker="o",
            label=extractor_key
        )

plt.xlabel("Top-k cutoff")
plt.ylabel("MRR@k")
plt.title("MRR Curves for Keyword Extraction Algorithms")
plt.xticks(range(1, 11))
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.show()
