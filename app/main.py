from fastapi import FastAPI, UploadFile, File
import pandas as pd

app = FastAPI(title="Wood AI CML ALO API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload-cml-data")
async def upload_cml_data(file: UploadFile = File(...)):
    df = pd.read_excel(file.file)
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
    }
