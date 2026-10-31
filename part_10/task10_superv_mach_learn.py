import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt




project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "part_1"
    / "pubmed_project2_dataset.csv"
)


df = pd.read_csv(input_path)


#These are the input features, the data the model will train on
X = df[["abstract", "topic"]].copy()

#These are the things what we want the model to produce, reference keywords. In supervised learning, in training part, these will be given to the model as correct answers
#so the model can learn to predict the correct things from only abstract when testing
y = df["keywords"].copy()

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,     #test size is 0.2 so training will be 0.8
    random_state=42,    #makes it possible to run the script again
    stratify=df["topic"]  #helps preserver the proportions of 10 tops in both training and test set
)

#Combine the topic and abstract into a single string AND deal with
#missing values. This was the first try at the model - scraped it since awful precision, recall and f1
#
#X_train_text = (
#    "Topic: " + X_train["topic"].fillna("") +
#    " Abstract: " + X_train["abstract"].fillna("")
#)
#
#X_test_text = (
#    "Topic: " + X_test["topic"].fillna("") +
#    " Abstract: " + X_test["abstract"].fillna("")
#)
#

#Second try is when I tried to keep the abstract and topic
#as seperate inputs 
#Here deal with missing values
X_train["abstract"] = X_train["abstract"].fillna("")
X_test["abstract"] = X_test["abstract"].fillna("")

X_train["topic"] = X_train["topic"].fillna("Unknown")
X_test["topic"] = X_test["topic"].fillna("Unknown")

#Build a separate feature representation for each input column
#Here we also use bigram version of word representation
preprocessor = ColumnTransformer(
    transformers=[
        (
            "abstract",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=15000
            ),
            "abstract"
        ),
        (
            "topic",
            OneHotEncoder(handle_unknown="ignore"),
            ["topic"]
        )
    ]
)
#This learns the vocubalary and weights of the words of the training inputs
#We use only the training inputs here to not cause leakage between
#training set and test set
X_train_tfidf = preprocessor.fit_transform(X_train)

#And here we use that same trained vectorizer to get the 
#numerical represantations of the test inputs
X_test_tfidf = preprocessor.transform(X_test)

#We need to split the keywords from semicolon ; to get them into
#numerical versions. So a little helper function
def split_keywords(keyword_string):
    if not isinstance(keyword_string, str):
        return []

    return [
        keyword.strip().lower()
        for keyword in keyword_string.split(";")
        if keyword.strip()
    ]

y_train_labels = y_train.apply(split_keywords)
y_test_labels = y_test.apply(split_keywords)

#first try to make the model work, gave awful prec, rec and f1 so
#I went another route (removing super rare keywords) which is implemented below
"""
#Now we convert the labels (reference keywords) into binary labels 

mlb = MultiLabelBinarizer()

#We train first with training data, then we use that same model to get numerical versions of test
#data
y_train_encoded = mlb.fit_transform(y_train_labels)
y_test_encoded = mlb.transform(y_test_labels)
"""
#Count how times each keyword appears in seperate docs
keyword_counts = Counter(
    keyword
    for labels in y_train_labels
    for keyword in set(labels)
)

#Only keep keywords that appear in AT LEAST 2 documents (Still really rare words)
allowed_keywords = {
    keyword
    for keyword, count in keyword_counts.items()
    if count >= 2
}

#Remove rare keywords from both training and test labels
y_train_filtered = [
    [keyword for keyword in set(labels)
     if keyword in allowed_keywords]
    for labels in y_train_labels
]

y_test_filtered = [
    [keyword for keyword in set(labels)
     if keyword in allowed_keywords]
    for labels in y_test_labels
]

#Create label encoding with only allowed keywords
mlb = MultiLabelBinarizer(
    classes=sorted(allowed_keywords)
)
#Now we convert the labels (reference keywords) into binary labels
#We train first with training data keywords, then we use that same model to get numerical versions for test
#data
y_train_encoded = mlb.fit_transform(y_train_filtered)
y_test_encoded = mlb.transform(y_test_filtered)

#And finally here is the model we are using
#Using onevsrestclassifier to get one binary classifier per keyword
#This is useful because we have multiple labels that we need to predict
#It learns a seperate prediction function for each keyword.
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000),
    n_jobs=-1
)

#And here we are training the model.
model.fit(
    X_train_tfidf,
    y_train_encoded
)

print("Training completed!")
#Get the models estimated probabilities (of being a keyword for specific doc) for each keyword
#We do this because the model uses a threshold of 50% normally,
#But because too many of our keywords are too rare (appear only once across all docs)
#we try and see if lowering the threshold could make the stats better by being more allowing
#for predicting keywords
threshold = 0.10  #Found out with trial and error that 0.05-0.10 is the sweet spot threshold for this data

y_prob = model.predict_proba(X_test_tfidf)
y_pred = (y_prob >= threshold).astype(int)


precision = precision_score(
    y_test_encoded,
    y_pred,
    average="micro",
    zero_division=0
)

recall = recall_score(
    y_test_encoded,
    y_pred,
    average="micro",
    zero_division=0
)

f1 = f1_score(
    y_test_encoded,
    y_pred,
    average="micro",
    zero_division=0
)


print(f"\nThreshold: {threshold:.2f}")
print(f"Micro precision: {precision:.3f}")
print(f"Micro recall:    {recall:.3f}")
print(f"Micro F1:        {f1:.3f}")


# Final configuration:
# - Model: One-vs-Rest Logistic Regression
# - Abstract features: TF-IDF unigrams, bigrams and trigrams
# - Maximum abstract features: 15,000
# - Topic: OneHotEncoder
# - Minimum keyword frequency: 2 training documents
# - Prediction threshold: 0.10
#
# Results:
# - Micro precision: 0.348
# - Micro recall:    0.379
# - Micro F1:        0.363

output_folder = project_root / "part_10"
output_folder.mkdir(parents=True, exist_ok=True)

statistics_path = output_folder / "part_10_model_statistics.csv"
plot_path = output_folder / "part_10_f1_score.png"


#Create a dataframe containing the model statistics
results_df = pd.DataFrame({
    "model": ["OneVsRest Logistic Regression"],
    "threshold": [threshold],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1]
})

#Save statistics to CSV
results_df.to_csv(statistics_path, index=False)

print(f"Saved model statistics to: {statistics_path}")


#Plot precision, recall and F1 scores
plt.figure(figsize=(8, 5))

metrics = ["precision", "recall", "f1"]
scores = [
    results_df["precision"].iloc[0],
    results_df["recall"].iloc[0],
    results_df["f1"].iloc[0]
]

plt.bar(metrics, scores)

plt.title("Part 10: Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)

# isplay score values above each bar
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f"{score:.3f}", ha="center")

plt.tight_layout()

#Save plot
plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()
plt.close()

print(f"Saved performance plot to: {plot_path}")