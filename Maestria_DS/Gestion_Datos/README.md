# Maestria_DS - Data Science Master's Program

A comprehensive Data Science project workspace for data management, analysis, and machine learning workflows.

## 📋 Project Overview

This project is designed as part of the Master's program in Data Science (Maestría en Data Science). It provides a structured environment for:
- Data exploration and analysis
- Statistical modeling
- Machine learning implementation
- Data visualization and reporting

## 📁 Project Structure

``
Maestria_DS/
├── data/                      # Data storage
│   ├── raw/                   # Original, immutable data
│   ├── processed/             # Cleaned, transformed data
│   └── external/              # External data sources
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/           # EDA and experimentation
│   ├── reports/               # Polished notebooks for analysis
│   └── tutorials/             # Learning materials
├── src/                       # Python modules
│   ├── data/                  # Data loading and processing
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and evaluation
│   └── visualization/         # Plotting and visualization utilities
├── reports/                   # Generated analysis and reports
│   ├── figures/               # Generated graphics
│   ├── tables/                # Analysis tables
│   └── README.md              # Report documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
``

## 🛠️ Installation

1. Clone the repository:
``ash
cd Maestria_DS
``

2. Create a virtual environment:
``ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``

3. Install dependencies:
``ash
pip install -r requirements.txt
``

4. Launch Jupyter Lab:
``ash
jupyter lab
``

## 📦 Key Libraries

- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Notebooks**: Jupyter, JupyterLab
- **Testing**: Pytest

## 🚀 Getting Started

1. Explore the 
otebooks/exploratory/ folder for example analyses
2. Check src/ for utility functions and modules
3. Add your data to data/raw/
4. Create new notebooks in 
otebooks/

## 📝 Usage

For data analysis:
``python
import pandas as pd
import numpy as np
from src.data import load_data
from src.visualization import plot_distribution

# Load data
data = load_data('data/raw/sample.csv')

# Analyze and visualize
print(data.describe())
plot_distribution(data)
``

## 🔧 Development

Run tests:
``ash
pytest tests/
``

Format code:
``ash
black src/ notebooks/
``

Lint code:
``ash
flake8 src/
``

## 📚 Documentation

- [Data Documentation](data/README.md) - Data dictionary and sources
- [Reports](reports/README.md) - Analysis results and findings
- [Code Examples](notebooks/tutorials/) - Learning materials

## 👤 Author

Christian Ruiz

## 📄 License

This project is part of the Master's program in Data Science.
