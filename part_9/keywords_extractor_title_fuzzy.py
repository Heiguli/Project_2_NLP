import pandas as pd
import json
from pathlib import Path

from part_2.rake_extractor import rake
from part_2.luhn_extractor import luhn
from part_2.yake_extractor import YAKE
from part_2.multiPartiteRank_extractor import get_multiPartiteResults
from part_2.positionRank_extractor import get_positionRank_results
from part_2.single_tpr import get_single_tpr_results
from part_2.singlerank import get_single_rank_results
from part_2.textrank import get_text_rank_results
from part_2.topicrank import get_topic_rank_results
from part_9.fuzzy_wuzzy_filtering import filter_keywords_by_title
from part_9.tf_tdf_extractor_title_fuzzy import get_idf_keywords
#Load the dataset
project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root / "part_1" /
    "pubmed_project2_dataset.csv"
)

output_path = (
    project_root / "part_9" /
    "pubmed_project2_with_keywords_title_fuzzy_wuzzy.csv"
)

df = pd.read_csv(input_path)

#requirements pip install fuzzywuzzy
#Connect each column name to its extractor function
extractors = {
    "rake_title_fuzzy_wuzzy": rake,
    "luhn_title_fuzzy_wuzzy": luhn,
    "yake_title_fuzzy_wuzzy": YAKE,
    "multipartiterank_title_fuzzy_wuzzy": get_multiPartiteResults,
    "positionrank_title_fuzzy_wuzzy": get_positionRank_results,
    "tpr_title_fuzzy_wuzzy": get_single_tpr_results,
    "singlerank_title_fuzzy_wuzzy": get_single_rank_results,
    "textrank_title_fuzzy_wuzzy": get_text_rank_results,
    "topicrank_title_fuzzy_wuzzy": get_topic_rank_results,
}


# 3. Run each extractor on every abstract
for column_name, extractor_function in extractors.items():

    print(f"Running {column_name}...")

    results = []

    for index, abstract in enumerate(df["abstract"]):

        try:
            keywords = extractor_function(abstract)
            title = df.loc[index, "title"]
            if pd.isna(title):
                title = ""
            # Make sure the result is a list
            if keywords is None:
                keywords = []

            # If the extractor returns (keyword, score) tuples,
            # keep only the keyword text.
            filtered_keywords = filter_keywords_by_title(title, keywords)
            results.append(json.dumps(filtered_keywords[:10], ensure_ascii=False))

        except Exception as error:
            print(f"Error in {column_name}, row {index}: {error}")
            results.append("[]")

    # 4. Save all results into the matching column
    df[column_name] = results

    # Save progress after each extractor, so completed work is preserved
    df.to_csv(output_path, index=False)

    print(f"Finished {column_name}!")

get_idf_keywords()
print(f"All extractors finished. Saved dataset to: {output_path}")






