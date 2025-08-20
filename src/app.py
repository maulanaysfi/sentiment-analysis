from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, os

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model = joblib.load('models/model.joblib')

class Query(BaseModel):
    text: str

@app.get('/')
def index():
    with open(f"{static_dir}/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
def predict(q: Query):
    pred = int(model.predict([q.text])[0])
    label = ['negative', 'neutral', 'positive'][pred]
    return {"label":label, "class_id":pred}