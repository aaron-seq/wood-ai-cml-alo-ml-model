# Wood AI CML ALO ML Model

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Machine Learning-powered **Condition Monitoring Location (CML) Optimization System** - A comprehensive data-driven solution for streamlining CML selection, elimination, and lifecycle management using advanced ML algorithms and industry best practices.

## Project Overview

This project implements an end-to-end ML system for CML optimization, originally developed by Wood PLC. The system combines machine learning, forecasting, and expert decision tracking to provide intelligent recommendations for CML elimination and inspection scheduling.

### Key Features

#### Core ML & Analytics
-   **Advanced ML Pipeline**: Random Forest classifier with hyperparameter tuning, cross-validation, and feature engineering
-   **Remaining Life Forecasting**: Time-series forecasting for remaining CML life and inspection scheduling
-   **Risk Classification**: Automated risk level assessment (CRITICAL, HIGH, MEDIUM, LOW)
-   **Feature Engineering**: 20+ engineered features including corrosion-thickness ratios and risk interactions
-   **Model Persistence**: Trained models saved with metadata and performance metrics

#### API & Integration
-   **FastAPI Backend**: High-performance RESTful API with OpenAPI documentation
-   **Comprehensive Endpoints**: Upload, score, forecast, SME overrides, and reporting
-   **Pydantic Validation**: Strict data validation with schemas for all inputs/outputs
-   **Batch Processing**: Support for bulk CML data processing and scoring
-   **File Format Support**: CSV and Excel (.xlsx) file uploads

#### Dashboard & Visualization
-   **Streamlit Dashboard**: Interactive web dashboard for data exploration and analysis
-   **Plotly Charts**: Dynamic visualizations for commodity distribution, risk analysis, and forecasts
-   **Real-time Statistics**: Live metrics and performance indicators
-   **Data Download**: Export predictions, forecasts, and reports as CSV

#### Expert Systems
-   **SME Override System**: Track Subject Matter Expert manual decision overrides
-   **Override Analytics**: Statistics on SME decisions and ML agreement rates
-   **Decision Tracking**: Complete audit trail of all manual interventions
-   **Reason Documentation**: Mandatory explanations for all override decisions

#### Production Ready
-   **Docker Support**: Full containerization with docker-compose
-   **Testing Suite**: Comprehensive pytest test coverage for API, ML, and utilities
-   **Documentation**: Complete API docs, usage guides, and deployment instructions
-   **Configuration Management**: Environment-based settings with .env support
-   **Logging & Monitoring**: Structured logging and health check endpoints

### Dataset

**Comprehensive 200-Row Synthetic Dataset** with realistic patterns:
- 200 unique CMLs (CML-001 to CML-200)
- 12 feature columns including corrosion rates, thickness, commodity types
- 9 commodity types: Crude Oil, Natural Gas, Steam, Fuel Gas, etc.
- 9 feature types: Pipe, Elbow, Tee, Flange, Reducer, Nozzle, Header, Bend, Weld
- 21% elimination rate (42 eliminate, 158 keep) - realistic industry ratio
- Risk scores, remaining life calculations, and inspection schedules

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/aaron-seq/wood-ai-cml-alo-ml-model.git
cd wood-ai-cml-alo-ml-model

# Start API server
docker-compose up --build

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/aaron-seq/wood-ai-cml-alo-ml-model.git
cd wood-ai-cml-alo-ml-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python ml/train_enhanced.py data/sample_cml_data.csv

# Start API server
uvicorn app.main:app --reload

# Start dashboard (separate terminal)
streamlit run streamlit_app.py
```

---

## Project Structure

```
wood-ai-cml-alo-ml-model/
├── app/                          # FastAPI application
│   ├── main.py                   # Main API with all endpoints
│   ├── main_enhanced.py          # Enhanced API with advanced features
│   ├── schemas.py                # Pydantic validation schemas
│   ├── config.py                 # Configuration and settings
│   ├── utils.py                  # Utility functions
│   ├── forecasting.py            # Remaining life forecasting module
│   ├── sme_override.py           # SME override management
│   └── requirements-api.txt      # API-specific dependencies
│
├── ml/                           # Machine Learning modules
│   ├── train_cml_model.py        # Basic training script
│   └── train_enhanced.py         # Advanced training with tuning
│
├── data/                         # Data directory
│   ├── sample_cml_data.csv       # 200-row comprehensive dataset
│   └── sme_overrides.json        # SME override records
│
├── models/                       # Trained model storage
│   └── cml_elimination_model.joblib  # Latest trained model
│
├── tests/                        # Test suite
│   ├── test_api.py               # API endpoint tests
│   ├── test_utils.py             # Utility function tests
│   └── test_forecasting.py       # Forecasting module tests
│
├── docs/                         # Documentation
│   ├── API_DOCUMENTATION.md      # Complete API reference
│   └── USAGE_GUIDE.md            # User guide and examples
│
├── streamlit_app.py              # Interactive dashboard
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
├── requirements.txt              # Python dependencies
├── requirements-streamlit.txt    # Dashboard dependencies
├── pytest.ini                    # Pytest configuration
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

---

## Usage

### 1. Training the Model

```bash
# Basic training
python ml/train_cml_model.py data/sample_cml_data.csv

# Enhanced training with hyperparameter tuning
python ml/train_enhanced.py data/sample_cml_data.csv
```

**Training Output:**
- Classification report with precision, recall, F1 scores
- ROC-AUC score and cross-validation results
- Feature importance rankings
- Model saved to `models/` directory

### 2. API Usage

#### Start the Server

```bash
uvicorn app.main:app --reload
```

API Documentation: http://localhost:8000/docs

#### Key Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Upload & Validate Data**
```bash
curl -X POST "http://localhost:8000/upload-cml-data" \
  -F "file=@data/sample_cml_data.csv"
```

**Score CML Data**
```bash
curl -X POST "http://localhost:8000/score-cml-data" \
  -F "file=@data/sample_cml_data.csv"
```

**Forecast Remaining Life**
```bash
curl -X POST "http://localhost:8000/forecast-remaining-life" \
  -F "file=@data/sample_cml_data.csv"
```

**Add SME Override**
```bash
curl -X POST "http://localhost:8000/sme-override" \
  -H "Content-Type: application/json" \
  -d '{
    "id_number": "CML-042",
    "sme_decision": "KEEP",
    "reason": "Critical monitoring point for high-risk area",
    "sme_name": "Dr. John Smith"
  }'
```

**Generate Comprehensive Report**
```bash
curl -X POST "http://localhost:8000/generate-report" \
  -F "file=@data/sample_cml_data.csv"
```

### 3. Dashboard Usage

```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

**Dashboard Features:**
-  **Overview**: Dataset statistics and visualizations
-  **Upload & Score**: Upload CML data for ML predictions
-  **Forecasting**: Generate remaining life forecasts and inspection schedules
-  **SME Overrides**: Manage expert manual overrides
-  **Reports**: Comprehensive analysis and downloadable reports

### 4. Python SDK Usage

```python
import pandas as pd
from app.forecasting import CMLForecaster
from app.sme_override import SMEOverrideManager

# Load data
df = pd.read_csv('data/sample_cml_data.csv')

# Forecast remaining life
forecaster = CMLForecaster(minimum_thickness=3.0, safety_factor=1.5)
forecast_df = forecaster.forecast_batch(df)
print(forecast_df[['id_number', 'remaining_life_years', 'risk_level']])

# Add SME override
sme_manager = SMEOverrideManager()
sme_manager.add_override(
    id_number='CML-042',
    sme_decision='KEEP',
    reason='Critical safety monitoring point',
    sme_name='Dr. Smith'
)

# Get override statistics
stats = sme_manager.get_override_statistics()
print(f"Total overrides: {stats['total_overrides']}")
```

---

##  Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=ml tests/

# Run specific test file
pytest tests/test_api.py -v

# Run only unit tests
pytest -m unit
```

**Test Coverage:**
- API endpoints (upload, score, forecast, SME)
- Utility functions (validation, scheduling)
- Forecasting module (calculations, risk levels)
- Data validation and error handling

---

##  Model Performance

### Classification Metrics (200-row dataset)

| Metric | Value |
|--------|-------|
| **Accuracy** | 90%+ |
| **Precision (Eliminate)** | 78% |
| **Recall (Eliminate)** | 70% |
| **F1 Score (Eliminate)** | 74% |
| **ROC-AUC** | 0.92+ |
| **Cross-validation F1** | 0.73 ± 0.08 |

### Feature Importance (Top 5)
1. `corrosion_thickness_ratio` (28%)
2. `average_corrosion_rate` (22%)
3. `thickness_mm` (18%)
4. `commodity` (15%)
5. `risk_score` (12%)

---

##   Business Value

### ROI Calculation
- **Investment**: $63K
- **Breakeven**: 6 clients
- **Target ROI**: 176%
- **Target EBITA**: 19% ($11,970 profit per client)

### Market Potential
- **Existing Clients**: ~10 (2 Canada, 4 Americas, 4 International)
- **Potential New Clients**: 6+ globally
- **CML Optimization Savings**: 20-40% reduction in monitoring costs
- **Inspection Efficiency**: 30% reduction in unnecessary inspections

---

##  Technology Stack

### Backend & ML
- **Python 3.9+**: Core programming language
- **FastAPI**: High-performance async API framework
- **scikit-learn**: Machine learning algorithms
- **pandas & numpy**: Data processing and analysis
- **joblib**: Model serialization

### Data & Validation
- **Pydantic**: Data validation and settings
- **openpyxl**: Excel file processing
- **python-multipart**: File upload handling

### Visualization & Dashboard
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Interactive visualizations
- **matplotlib & seaborn**: Statistical plots

### Testing & DevOps
- **pytest**: Testing framework
- **Docker & Docker Compose**: Containerization
- **uvicorn**: ASGI server

---

##  Documentation

Comprehensive documentation available in `/docs` directory:

- **[API Documentation](docs/API_DOCUMENTATION.md)**: Complete API reference with examples
- **[Usage Guide](docs/USAGE_GUIDE.md)**: Step-by-step usage instructions and best practices

---

##  Roadmap

###   Phase 1 - Core ML System (Completed)
- [x] 200-row comprehensive synthetic dataset
- [x] Enhanced ML training with hyperparameter tuning
- [x] Forecasting module for remaining life
- [x] SME override system
- [x] Pydantic schemas and validation
- [x] Comprehensive API endpoints
- [x] Streamlit dashboard
- [x] Testing suite
- [x] Docker deployment
- [x] Complete documentation

###  Phase 2 - Advanced Features (In Progress)
- [ ] PDF report generation with charts
- [ ] Time-series analysis with historical data
- [ ] Advanced anomaly detection
- [ ] Multi-model ensemble predictions
- [ ] Real-time monitoring dashboard

###  Phase 3 - Enterprise Integration (Planned)
- [ ] Microsoft Azure cloud deployment
- [ ] PostgreSQL database integration
- [ ] User authentication and authorization
- [ ] API rate limiting and caching
- [ ] Automated CI/CD pipeline
- [ ] Integration with existing Wood systems

###  Phase 4 - AI Enhancement (Future)
- [ ] Deep learning models for complex patterns
- [ ] NLP for SME reason analysis
- [ ] Automated report generation with GPT
- [ ] Predictive maintenance scheduling
- [ ] Smart recommendations engine

---

##  Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Guidelines:**
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

---

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

##  Team

**Original Project (Wood PLC):**
- **Project Owner**: Jeffrey Anokye
- **Development Lead**: Jason Strouse
- **Project Lead**: Mariana Lima

**Current Development:**
- **Developer**: Aaron Sequeira (Smarter.Codes.AI)
- **Organization**: Smarter.Codes.AI
- **Duration**: December 2024

---

##  Contact & Support

- **Developer**: Aaron Sequeira
- **Email**: aaron@smarter.codes.ai
- **GitHub Issues**: [Create Issue](https://github.com/aaron-seq/wood-ai-cml-alo-ml-model/issues)
- **Documentation**: See `/docs` directory

---

##  Acknowledgments

- Wood PLC for the original project concept and funding
- Subject Matter Experts who provided domain knowledge
- scikit-learn and FastAPI communities
- Open-source ML and data science ecosystem

---

##  Project Stats

![GitHub Stars](https://img.shields.io/github/stars/aaron-seq/wood-ai-cml-alo-ml-model?style=social)
![GitHub Forks](https://img.shields.io/github/forks/aaron-seq/wood-ai-cml-alo-ml-model?style=social)
![GitHub Issues](https://img.shields.io/github/issues/aaron-seq/wood-ai-cml-alo-ml-model)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/aaron-seq/wood-ai-cml-alo-ml-model)

---

**Made with ❤️ by Aaron Sequeira**  
**Original Concept by Wood PLC Engineering Team**
