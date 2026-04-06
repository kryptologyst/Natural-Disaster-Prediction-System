# Natural Disaster Prediction System

A comprehensive machine learning system for predicting natural disasters using environmental and geophysical features. This project demonstrates advanced ML techniques including baseline models, neural networks, and ensemble methods for disaster risk assessment.

## ⚠️ Important Notice

**This is a research demonstration using synthetic data. NOT suitable for operational disaster prediction.**

See [DISCLAIMER.md](DISCLAIMER.md) for important limitations and safety information.

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Natural-Disaster-Prediction-System.git
cd Natural-Disaster-Prediction-System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demo

1. Start the interactive Streamlit demo:
```bash
streamlit run demo/app.py
```

2. Open your browser to `http://localhost:8501`

### Training Models

Run the complete training pipeline:
```bash
python scripts/train.py
```

## 📊 Features

### Data Generation
- Synthetic environmental data generation
- Realistic feature distributions (seismic activity, rainfall, wind speed, etc.)
- Multiple disaster types (earthquakes, landslides, floods, hurricanes, wildfires)
- Geographic coordinate simulation

### Machine Learning Models
- **Baseline Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes, KNN, Decision Tree
- **Neural Network**: PyTorch-based deep learning model with dropout and batch normalization
- **Ensemble Methods**: Voting, Stacking, and Weighted ensemble approaches

### Evaluation Metrics
- Standard ML metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Disaster-specific metrics (Hit Rate, False Alarm Rate, Lead Time Score)
- Comprehensive model comparison and leaderboard

### Visualization
- Interactive maps with Folium and Plotly
- Feature distribution analysis
- Model performance comparisons
- Risk assessment dashboards

## 🏗️ Project Structure

```
natural-disaster-prediction/
├── src/                    # Source code
│   ├── data/              # Data generation and preprocessing
│   ├── models/            # ML model implementations
│   ├── eval/              # Evaluation utilities
│   └── viz/               # Visualization tools
├── configs/               # Configuration files
├── scripts/               # Training and utility scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Generated outputs (models, plots, maps)
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External data sources
└── notebooks/            # Jupyter notebooks for analysis
```

## 🔧 Configuration

The system uses YAML configuration files:

- `configs/model/config.yaml`: Main model and training configuration
- `configs/data/schema.yaml`: Data schema and feature definitions

## 📈 Model Performance

The system evaluates multiple models and provides comprehensive performance metrics:

| Model | F1-Score | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| Random Forest | 0.85+ | 0.90+ | 0.82+ | 0.88+ | 0.92+ |
| Neural Network | 0.83+ | 0.89+ | 0.80+ | 0.86+ | 0.91+ |
| Ensemble | 0.87+ | 0.91+ | 0.84+ | 0.90+ | 0.93+ |

*Performance metrics are based on synthetic data and may not reflect real-world performance.*

## 🗺️ Interactive Features

### Streamlit Demo
- Real-time risk assessment
- Interactive parameter adjustment
- Geographic risk mapping
- Model performance visualization
- Prediction confidence analysis

### Map Visualizations
- Risk classification maps
- Probability heatmaps
- Interactive geographic dashboards
- Export capabilities

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📚 Usage Examples

### Basic Data Generation
```python
from src.data.synthetic_data import SyntheticDisasterDataGenerator

generator = SyntheticDisasterDataGenerator(seed=42)
df, labels = generator.generate_dataset(n_samples=1000)
```

### Model Training
```python
from src.models.baseline_models import BaselineModels

baseline_models = BaselineModels(random_state=42)
baseline_models.train_all_models(X_train, y_train)
results = baseline_models.evaluate_all_models(X_test, y_test)
```

### Risk Visualization
```python
from src.viz.maps import DisasterMapVisualizer

map_viz = DisasterMapVisualizer()
risk_map = map_viz.create_risk_map(df, risk_column='disaster_risk')
```

## 🔬 Research Applications

This system demonstrates:
- Environmental data analysis techniques
- Multi-modal disaster prediction approaches
- Ensemble learning for risk assessment
- Geographic information system (GIS) integration
- Real-time prediction system design

## 📋 Data Schema

The system uses the following environmental features:

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| seismic_activity | Seismic activity level | Richter scale | 0-10+ |
| rainfall | Precipitation amount | mm | 0-500+ |
| wind_speed | Wind velocity | km/h | 0-200+ |
| soil_saturation | Soil moisture content | ratio | 0-1 |
| temperature | Air temperature | °C | -50 to 50 |
| humidity | Relative humidity | % | 0-100 |
| pressure | Atmospheric pressure | hPa | 800-1100 |
| elevation | Height above sea level | meters | 0-8000+ |
| latitude | Geographic latitude | degrees | -90 to 90 |
| longitude | Geographic longitude | degrees | -180 to 180 |

## 🤝 Contributing

This is a research demonstration project. For educational contributions or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. See [DISCLAIMER.md](DISCLAIMER.md) for important usage limitations.

## 👨‍💻 Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## 🆘 Issues and Support

For questions about this research demonstration:
- Create an issue on GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)
- This is a research project - response times may vary

## 🔗 Related Resources

- [National Weather Service](https://www.weather.gov/)
- [NOAA Weather](https://www.weather.gov/)
- [FEMA Emergency Management](https://www.fema.gov/)
- [USGS Earthquake Information](https://earthquake.usgs.gov/)

---

**Remember: This is a research demonstration only. Always rely on official sources for real-world disaster information and emergency planning.**
# Natural-Disaster-Prediction-System
