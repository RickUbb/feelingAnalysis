from fastapi import FastAPI
import model

app = FastAPI()

@app.get("/")
def root():
    return { "ok": 1 }

@app.get("/analyze")
def analyze ( q: str ):
    return model.analiza( q )
