from Bio import Entrez, Medline
import pandas as pd
from pathlib import Path

Entrez.email = "hseikki@hotmail.com"

suffix = ' AND 2020:2025[dp] AND english[la] AND hasabstract AND "journal article"[pt] NOT review[pt]'

queries = {
    "Type 2 diabetes": '"Diabetes Mellitus, Type 2"[MeSH Major Topic] OR "type 2 diabetes"[Title]',
    "Breast cancer": '"Breast Neoplasms"[MeSH Major Topic] OR "breast cancer"[Title]',
    "Alzheimer's disease": '"Alzheimer Disease"[MeSH Major Topic] OR "Alzheimer"[Title]',
    "COVID-19": '"COVID-19"[MeSH Major Topic] OR "COVID-19"[Title]',
    "Depression": '"Depressive Disorder, Major"[MeSH Major Topic] OR "major depressive disorder"[Title]',
    "Hypertension": '"Hypertension"[MeSH Major Topic] OR "hypertension"[Title]',
    "Asthma": '"Asthma"[MeSH Major Topic] OR "asthma"[Title]',
    "Inflammatory bowel disease": '"Inflammatory Bowel Diseases"[MeSH Major Topic] OR "inflammatory bowel disease"[Title]',
    "Osteoarthritis": '"Osteoarthritis"[MeSH Major Topic] OR "osteoarthritis"[Title]',
    "Antimicrobial resistance": '"Drug Resistance, Bacterial"[MeSH Major Topic] OR "antimicrobial resistance"[Title]',
}

seen_articles = set()

#Get the ids based on queries and suffix
#Each topic gets 400 ids to make sure we get enough
#workable records
#Using library Entrez from biopython
def get_ids():
    ids_by_topic = {}
    for topic, query in queries.items():
        handle = Entrez.esearch(
            db="pubmed",
            term=f"({query})" + suffix,
            retmax=400,
            sort="pub_date",
        )
        record = Entrez.read(handle)
        ids_by_topic[topic] = record["IdList"]
        handle.close()
    return ids_by_topic



ids_by_topic = get_ids()

#Then we cycle through those ids and get workable articles 
#We store pmid, topic, title, abstract and keywordws of each 
#record that has required data
def get_article_datas():
    papers = []
    for topic_name, ids in ids_by_topic.items():
        rows = []
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        for record in records:
            pmid = record.get("PMID")
            title, abstract, keywords = record.get("TI"), record.get("AB"), record.get("OT")
            if pmid in seen_articles or not (title and abstract and keywords):
                continue
            seen_articles.add(pmid)
            rows.append({
                "pmid": pmid,
                "topic": topic_name,
                "title": title,
                "abstract": abstract,
                "keywords": "; ".join(k.strip() for k in keywords),
            })
            if len(rows) == 100:
                break
        print(f"{topic_name}: kept {len(rows)}")
        papers.extend(rows)
    return papers

#With pandas we save the list of rows (row is one article with pmid, topic, title, abs, kws)
#into a csv file
df = pd.DataFrame(get_article_datas())
script_folder = Path(__file__).resolve().parent
output_path = script_folder / "pubmed_project2_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
print(df.shape)  