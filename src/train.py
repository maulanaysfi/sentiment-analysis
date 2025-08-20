import logging
import mlflow
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from .preprocess import load_sentiments, split

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info("Importing dependencies... DONE")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
logger.info(f"Setting MLflow tracking URI to {MLFLOW_TRACKING_URI}...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info("Tracking URI set successfully.")

# Load dataset
data_path = "data/training.1600000.processed.noemoticon.csv"
logger.info(f"Loading dataset from '{data_path}' and splitting...")

df = load_sentiments(data_path)
X_train, X_test, y_train, y_test = split(df)
logger.info("Dataset loaded and split successfully.")

# Define pipeline
logger.info("Building model pipeline...")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
])
logger.info("Pipeline created successfully.")

# Train and log with MLflow
logger.info("Starting MLflow run and training the model...")
model_path = "models/model.joblib"

with mlflow.start_run():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_text(classification_report(y_test, y_pred), "cls_report.txt")

    joblib.dump(pipe, model_path)
    mlflow.log_artifact(model_path)

logger.info(f"Model training complete. Saved model at {model_path}")
