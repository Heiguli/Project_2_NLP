
import json
from pathlib import Path
import pandas as pd
from part_3.keyword_statistics_extractor import get_keyword_extraction_statistics
import matplotlib.pyplot as plt
project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root / "part_2" / "pubmed_project2_with_keywords.csv"
)

output_path = (
    project_root / "part_3" / "pubmed_project2_with_keywords_with_statistics.csv"
)

df = pd.read_csv(input_path)


extractor_keys = [
    'rake','luhn','yake','multipartiterank','positionrank','tpr','singlerank','textrank','topicrank', 'tfidf'
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

#Calculate the mean scores for each algorithm across topics
means = (
    results_df.groupby("extractor")[["precision", "recall", "f1"]]
    .mean()
    .sort_values("f1", ascending=True)
)


#Mean F1 score graph

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(means.index, means["f1"])

ax.bar_label(bars, fmt="%.3f", padding=4)
ax.set_title("F1 statistic by Keyword Extraction Algorithm")
ax.set_xlabel("F1")
ax.set_ylabel("Algorithm")
ax.set_xlim(0, means["f1"].max() * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()

f1_plot_path = project_root / "part_3" / "mean_f1_by_algorithm.png"

plt.savefig(f1_plot_path, dpi=300, bbox_inches="tight")
plt.show()


#Mean prec, recall and f1 graph

plot_data = means.sort_values("f1", ascending=False)
ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

ax.set_title("Precision, Recall and F1 by Algorithm")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Scores")
ax.tick_params(axis="x", rotation=40)
ax.legend(title="Metric")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()

f1_plot_path2 = project_root / "part_3" / "mean_precision_recall_f1.png"

plt.savefig(
    f1_plot_path2,
    dpi=300,
    bbox_inches="tight"
)
plt.show()
