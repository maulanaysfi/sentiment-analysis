import re, pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# cleaning tweets with regexp
def clean_tweet(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " URL ", t)
    t = re.sub(r"@\w+", " USER ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# load raw data
def load_sentiments(path):
    cols = ["label","id","date","query","user","tweet"]
    df = pd.read_csv(path, encoding="latin-1", names=cols)
    df["label"] = df["label"].map({0:0, 2:1, 4:2})
    df["tweet"] = df["tweet"].astype(str).map(clean_tweet)
    return df[["tweet","label"]].dropna()

def split(df):
    return train_test_split(df["tweet"], df["label"], test_size=0.2, random_state=33, stratify=df["label"])

