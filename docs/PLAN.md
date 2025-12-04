# Wood AI CML ALO ML Model

Initial API + Docker skeleton added:
- FastAPI app with /health and /upload-cml-data endpoints
- Dockerfile and docker-compose.yml for local testing
- Synthetic sample CML dataset at data/sample_cml_data.csv

Next steps (to be implemented):
- Full ML pipeline for CML elimination recommendation
- Forecasting module for corrosion and remaining life
- SME override and feedback loop
- Dashboard (likely with Streamlit or similar)
- PDF export of summary report
