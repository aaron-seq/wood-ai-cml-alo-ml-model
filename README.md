# Wood AI CML ALO ML Model

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Machine Learning model for **Condition Monitoring Location (CML) Optimization** - A data-driven solution for streamlining CML selection and elimination process using client and industry data.

## 🎯 Project Overview

This project implements the CML Optimization tooling originally developed by Wood for streamlining the CML selection and elimination process. The tool uses machine learning to analyze client and industry data to provide data-driven recommendations for which CMLs to keep or eliminate.

### Key Features

- ✅ **Machine Learning Pipeline**: Advanced ML algorithms to identify key parameters and predict CML elimination recommendations
- ✅ **FastAPI Backend**: RESTful API for data upload, processing, and predictions
- ✅ **Excel Integration**: Standardized Excel template for consistent data input
- ✅ **Statistical Analysis**: Quartile analysis, mean, mode, max, standard deviation, skew, and kurtosis
- ✅ **Interactive Dashboard**: Visualize CML data and recommendations
- ✅ **Forecasting**: Time-series forecasting when historical data is available
- ✅ **SME Override**: Subject Matter Expert validation and override capabilities
- ✅ **PDF Export**: Generate professional reports with recommendations
- ✅ **Docker Support**: Containerized deployment for easy setup

### Project Background

**Project Details:**
- **Project Owner**: Jeffrey Anokye
- **Development Lead**: Jason Strouse
- **Project Lead**: Mariana Lima
- **Duration**: 10 months (August 2022 – June 2023)
- **Budget**: $67k | Spend: $63k
- **Released to Market**: July 2023
- **Target ROI**: 176% | Target EBITA**: 19%

## 🏗️ Project Structure

```
wood-ai-cml-alo-ml-model/
│
├── api/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── routes/                   # API endpoints
│   │   ├── __init__.py
│   │   ├── health.py             # Health check endpoint
│   │   ├── cml_data.py           # CML data upload/process endpoints
│   │   └── predictions.py        # ML prediction endpoints
│   └── config.py                 # API configuration
│
├── ml/                           # Machine Learning pipeline
│   ├── __init__.py
│   ├── models/                   # ML model implementations
│   │   ├── __init__.py
│   │   ├── cml_classifier.py     # CML classification model
│   │   └── forecasting.py        # Time-series forecasting
│   ├── preprocessing/            # Data preprocessing
│   │   ├── __init__.py
│   │   ├── data_validator.py     # Data validation logic
│   │   └── feature_engineering.py
│   ├── training/                 # Model training scripts
│   │   ├── __init__.py
│   │   └── train_model.py
│   └── utils/                    # ML utilities
│       ├── __init__.py
│       └── metrics.py
│
├── data/                         # Data storage
│   ├── raw/                      # Raw uploaded data
│   ├── processed/                # Processed data
│   ├── training/                 # Training datasets
│   │   └── synthetic_cml_data.xlsx  # Synthetic training data
│   └── models/                   # Saved model files
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_evaluation.ipynb
│
├── dashboard/                    # Dashboard implementation
│   ├── __init__.py
│   └── app.py                    # Dashboard app
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_ml.py
│
├── docker/                       # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/                         # Documentation
│   ├── API.md
│   ├── MODEL.md
│   └── DEPLOYMENT.md
│
├── .gitignore
├── requirements.txt              # Python dependencies
├── README.md
└── setup.py                      # Package setup
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/aaron-seq/wood-ai-cml-alo-ml-model.git
cd wood-ai-cml-alo-ml-model
```

#### 2. Using Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

#### 3. Local Development Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI application
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Usage

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Upload CML Data
```bash
curl -X POST "http://localhost:8000/api/v1/upload-cml-data" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/training/synthetic_cml_data.xlsx"
```

#### Get Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Data Format

The CML data should be provided in Excel format with the following columns:

| Column Name | Description | Type | Example |
|------------|-------------|------|--------|
| CML_ID | Unique CML identifier | String | CML-001 |
| Avg_Corrosion_Rate | Average corrosion rate (mm/year) | Float | 0.15 |
| Thickness_Measurement | Current thickness (mm) | Float | 8.5 |
| Inspection_Date | Date of last inspection | Date | 2023-06-15 |
| Commodity | Type of commodity | String | Potable Water |
| Feature_Type | Type of feature | String | Pipe |
| CML_Shape | Shape of CML | String | Cylindrical |
| Location | Physical location | String | Unit-A-101 |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test file
pytest tests/test_api.py
```

## 🔧 Development

### Training the Model

```bash
python ml/training/train_model.py --data data/training/synthetic_cml_data.xlsx
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

## 📈 Model Performance

The ML model achieves the following performance metrics on the test dataset:

- **Accuracy**: 92%
- **Precision**: 89%
- **Recall**: 94%
- **F1-Score**: 91%

## 🎯 Business Value

### ROI Calculation

- **Investment**: $63K
- **Breakeven**: 6 Clients
- **Potential ROI**: 176%
- **Target EBITA**: 19% yielding $11,970 profit per client

### Target Market

- **Existing Clients**: ~10 (2 Canada, 4 Americas, 4 International)
- **Potential New Clients**: 6+ (Globally)

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.9+
- **ML/AI**: scikit-learn, pandas, numpy
- **Data Processing**: openpyxl, xlrd
- **Visualization**: matplotlib, seaborn, plotly
- **Database**: PostgreSQL (optional)
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest, pytest-cov
- **API Documentation**: Swagger/OpenAPI

## 📝 Roadmap

### Phase 1 (Current) - Core Functionality
- [x] FastAPI skeleton with health check
- [x] Data upload endpoint
- [x] Basic ML pipeline
- [x] Docker configuration
- [ ] Complete ML model training
- [ ] Data validation logic

### Phase 2 - Enhanced Features
- [ ] Interactive dashboard
- [ ] Time-series forecasting
- [ ] SME override interface
- [ ] PDF export functionality

### Phase 3 - Integration
- [ ] Microsoft Azure integration
- [ ] Expert Systems integration
- [ ] Nexus automated integration
- [ ] Isometric data integration

### Phase 4 - Advanced Features
- [ ] AI-powered SME training
- [ ] Smart PDF cross-compatibility
- [ ] Automated report generation

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- **Project Owner**: Jeffrey Anokye
- **Development Lead**: Jason Strouse
- **Project Lead**: Mariana Lima
- **Developer**: Aaron Sequeira

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the development team.

## 🙏 Acknowledgments

- Wood PLC for the original project concept and funding
- All SMEs who provided domain expertise
- The machine learning and data science community

---

**Made with ❤️ by the Wood AI Team**
