
import json
from pathlib import Path
import pandas as pd

from part_2.rake_extractor import rake
from part_2.luhn_extractor import luhn
from part_2.yake_extractor import YAKE
from part_2.multiPartiteRank_extractor import get_multiPartiteResults
from part_2.positionRank_extractor import get_positionRank_results
from part_2.single_tpr import get_single_tpr_results
from part_2.singlerank import get_single_rank_results
from part_2.textrank import get_text_rank_results
from part_2.topicrank import get_topic_rank_results
from part_2.tf_idf_extractor import get_idf_keywords

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root / "part_1" / "pubmed_project2_dataset.csv"
)

output_path = (
    project_root / "part_2" / "pubmed_project2_with_keywords.csv"
)

df = pd.read_csv(input_path)


#Extractor names connected to the function names
extractors = {
    "rake": rake,
    "luhn": luhn,
    "yake": YAKE,
    "multipartiterank": get_multiPartiteResults,
    "positionrank": get_positionRank_results,
    "tpr": get_single_tpr_results,
    "singlerank": get_single_rank_results,
    "textrank": get_text_rank_results,
    "topicrank": get_topic_rank_results,
}

#Run each extractor on every abstract
for column_name, extractor_function in extractors.items():

    print(f"Running {column_name}...")

    results = []

    for index, abstract in enumerate(df["abstract"]):

        try:
            keywords = extractor_function(abstract)
            if keywords is None:
                keywords = []
            results.append(json.dumps(keywords[:10], ensure_ascii=False))

        except Exception as error:
            print(f"Error in {column_name}, row {index}: {error}")
            results.append("[]")

    #Save all results into the matching column
    df[column_name] = results

    #Save progress after each extractor, so completed work is preserved
    df.to_csv(output_path, index=False)

    print(f"Finished {column_name}!")

get_idf_keywords()
print(f"All extractors finished. Saved dataset to: {output_path}")






