from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('models/model.joblib')

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(q: Query):
    pred = int(model.predict([q.text])[0])
    label = ['negative', 'neutral', 'positive']
    return {"label":label, "class_id":pred}