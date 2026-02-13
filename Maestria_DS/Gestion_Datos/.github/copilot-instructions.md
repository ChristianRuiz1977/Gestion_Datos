# Maestria_DS - Data Science Project

This is a Data Science Master's program workspace with a complete project structure for data analysis, visualization, and machine learning.

## Project Setup Completed

Project structure has been successfully scaffolded with the following components:

### Directory Structure
- **data/**: Data storage (raw, processed, external)
- **notebooks/**: Jupyter notebooks (exploratory, reports, tutorials)
- **src/**: Python modules (data, features, models, visualization)
- **reports/**: Generated outputs (figures, tables)
- **.vscode/**: IDE configuration
- **.github/**: GitHub configuration

### Installed Modules
- src/data/loader.py: Data loading utilities
- src/visualization/plots.py: Visualization helpers
- Python package structure with __init__.py files

### Configuration Files
- equirements.txt: Python dependencies (numpy, pandas, scikit-learn, matplotlib, jupyter, etc.)
- .gitignore: Git ignore rules for Python projects
- README.md: Project documentation
- .vscode/settings.json: VS Code settings for Python development

## Next Steps

1. **Create Virtual Environment**:
   `ash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   `

2. **Install Dependencies**:
   `ash
   pip install -r requirements.txt
   `

3. **Start Jupyter Lab**:
   `ash
   jupyter lab
   `

4. **Begin Data Analysis**:
   - Add your data to data/raw/
   - Create notebooks in 
otebooks/exploratory/
   - Use utilities from src/ modules

## Development Guidelines

- Use Python 3.9+
- Follow PEP 8 code style
- Add docstrings to functions and modules
- Create reusable utility functions in src/
- Document your analysis in notebooks
- Use version control for tracking changes
