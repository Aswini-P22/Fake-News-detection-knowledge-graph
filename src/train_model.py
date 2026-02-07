import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

#MODELS TO COMPARE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# 1. Load cleaned dataset

DATA_PATH = r"C:\Users\Praveen\OneDrive\Desktop\fake news detection\data\preprocessed data\cleaned_fakenews.csv"
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded:", df.shape)

X = df["final_text"]         
y = df["target"]              

# 2. Train-test split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )


# 3. TF-IDF Vectorization

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 4. Models + Hyperparameters

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 20]}
    ),
    "Linear SVM": (
        LinearSVC(),
        {"C": [0.5, 1]}
    ),
    "Multinomial NB": (
        MultinomialNB(),
        {"alpha": [0.5, 1.0]}
    )
}


# 5. Model comparison

results = []
best_model = None
best_f1 = 0

for name, (model, params) in models.items():
    print(f"\n🔹 Training {name}")

    grid = GridSearchCV(
        model, params,
        scoring="f1",  cv=3, n_jobs=-1
    )

    grid.fit(X_train_vec, y_train)

    y_pred = grid.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "Best Params": grid.best_params_
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = grid.best_estimator_

# 6. Results

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n📊 MODEL COMPARISON")
print(results_df)


# 7. Save best model

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Best model saved")
print("🏆 Selected model:", results_df.iloc[0]["Model"])