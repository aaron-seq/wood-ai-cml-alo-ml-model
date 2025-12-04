# Quick Start Guide

## 5-Minute Setup

### Prerequisites
```bash
# Check Python version (3.9+ required)
python --version

# Check Docker (optional)
docker --version
```

### Option 1: Local Setup (Fastest)

```bash
# 1. Clone and navigate
git clone https://github.com/aaron-seq/wood-ai-cml-alo-ml-model.git
cd wood-ai-cml-alo-ml-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (uses sample data)
python ml/train_enhanced.py data/sample_cml_data.csv

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open in browser
# API Docs: http://localhost:8000/docs
# Health Check: http://localhost:8000/health
```

### Option 2: Docker Setup

```bash
# 1. Clone
git clone https://github.com/aaron-seq/wood-ai-cml-alo-ml-model.git
cd wood-ai-cml-alo-ml-model

# 2. Start with Docker
docker-compose up --build

# 3. Access
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## Test the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload and validate data
curl -X POST "http://localhost:8000/upload-cml-data" \
  -F "file=@data/sample_cml_data.csv"

# Get predictions
curl -X POST "http://localhost:8000/score-cml-data" \
  -F "file=@data/sample_cml_data.csv"

# Forecast remaining life
curl -X POST "http://localhost:8000/forecast-remaining-life" \
  -F "file=@data/sample_cml_data.csv"
```

### Using Python Requests

```python
import requests

# Score CML data
with open('data/sample_cml_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/score-cml-data',
        files={'file': f}
    )
    print(response.json())
```

### Using the Interactive Docs

1. Open http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Upload `data/sample_cml_data.csv`
5. Click "Execute"

---

## Launch the Dashboard

```bash
# Install dashboard dependencies
pip install -r requirements-streamlit.txt

# Launch dashboard
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

**Dashboard Features:**
- Upload and visualize CML data
- Get ML predictions
- Generate forecasts
- Manage SME overrides
- Download reports

---

## Run Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=ml tests/

# Run specific tests
pytest tests/test_api.py -v
```

---

## Common Tasks

### 1. Train with Your Own Data

```bash
# Prepare your CSV with required columns:
# - id_number
# - average_corrosion_rate
# - thickness_mm
# - commodity
# - feature_type
# - cml_shape
# - elimination_flag (0 or 1)

python ml/train_enhanced.py path/to/your/data.csv
```

### 2. Add SME Override

```bash
curl -X POST "http://localhost:8000/sme-override" \
  -H "Content-Type: application/json" \
  -d '{
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "Critical safety monitoring point",
    "sme_name": "Dr. John Smith"
  }'
```

### 3. Generate Report

```bash
curl -X POST "http://localhost:8000/generate-report" \
  -F "file=@data/sample_cml_data.csv" \
  > report.json
```

### 4. Use Python SDK

```python
from app.forecasting import CMLForecaster
import pandas as pd

# Load data
df = pd.read_csv('data/sample_cml_data.csv')

# Forecast
forecaster = CMLForecaster()
results = forecaster.forecast_batch(df)

# View results
print(results[['id_number', 'remaining_life_years', 'risk_level']])
```

---

## Troubleshooting

### Port Already in Use
```bash
# Use different port
uvicorn app.main:app --port 8001
```

### Model Not Found
```bash
# Train the model first
python ml/train_enhanced.py data/sample_cml_data.csv
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Import Errors
```bash
# Make sure you're in the project root
pwd  # Should show .../wood-ai-cml-alo-ml-model

# Set PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Next Steps

1. **Read Full Documentation**
   - [README.md](README.md)
   - [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
   - [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)

2. **Explore the Code**
   - Check `app/main.py` for API endpoints
   - Review `ml/train_enhanced.py` for ML pipeline
   - Examine `streamlit_app.py` for dashboard

3. **Customize**
   - Update `.env` for your configuration
   - Modify thresholds in `app/config.py`
   - Add your own data and retrain

4. **Deploy**
   - Use Docker for production
   - Set up CI/CD pipeline
   - Configure monitoring

---

## Getting Help

- **Documentation**: Check `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/aaron-seq/wood-ai-cml-alo-ml-model/issues)
- **Email**: aaron@smarter.codes.ai

---

**You're all set! 🎉**

The system is ready to use with the included 200-row sample dataset.  
For production use, replace with your actual CML data and retrain the model.

*Happy optimizing!*