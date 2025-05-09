from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.textSummarizer.pipeline.prediction import PredictionPipeline

text: str = "What is Text Summarization"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.post("/predict")
async def predict_route(text: str, summary_type: str = "short"):
    try:
        obj = PredictionPipeline()
        # Pass the summary_type to the predict method
        summary = obj.predict(text, summary_type)
        return {"summary": summary}
    except Exception as e:
        raise e
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
