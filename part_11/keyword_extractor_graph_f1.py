
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#Load the results
project_root = Path(__file__).resolve().parent.parent
part_11_folder = project_root / "part_11"

#Load the results from Part 11
input_path = (
    part_11_folder
    / "all_keyword_extraction_method_statistics_by_method.csv"
)

df = pd.read_csv(input_path)


#Calculate the mean scores for each algorithm across topics
means = (
    df.groupby("extractor")[["precision", "recall", "f1"]]
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
plt.savefig(project_root / "part_11" / "mean_f1_by_algorithm.png", dpi=300, bbox_inches="tight")
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
plt.savefig(
    project_root / "part_11"/ "precision_recall_f1_of_algorithms",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

