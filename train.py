import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading datasets...")

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Assign labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# Remove empty articles
data = data[data["content"].str.strip() != ""]

# Remove duplicate articles
data = data.drop_duplicates(subset="content")

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

print("Total articles:", len(data))

# Features and labels
X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Fake News", "Real News"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")

print("\nModel saved successfully!")
print("Location: model/fake_news_model.pkl")
