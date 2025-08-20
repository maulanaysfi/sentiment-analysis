# building baseline model
print("Importing dependencies...", end="")
import mlflow, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from .preprocess import load_sentiments, split
print("DONE!")

print("Setting MLflow artifact URI path...", end="")
mlflow.set_tracking_uri("http://localhost:5000")
print("DONE!")

print("Loading dataset and splitting...", end="")
data_path = 'data/training.1600000.processed.noemoticon.csv'

df = load_sentiments(data_path)
X_train, X_test, y_train, y_test = split(df)
print("DONE!")

print("Running model pipeline...", end="")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
])

with mlflow.start_run():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_text(classification_report(y_test, y_pred), "cls_report.txt")
    joblib.dump(pipe, "models/model.joblib")
    mlflow.log_artifact("models/model.joblib")

print("DONE!")
print("Saved models/model.joblib")