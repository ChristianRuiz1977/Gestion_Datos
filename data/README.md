# Data Directory

## Overview

This directory contains all data files for the Maestria_DS project.

### Subdirectories

#### raw/
Original, immutable data files. Never edit files in this directory.
- Source data from external databases or files
- Original data format without any modifications

#### processed/
Cleaned and transformed data ready for analysis.
- Result of data cleaning and preprocessing
- Intermediate datasets for modeling
- Normalized and formatted data

#### external/
External data sources from public or commercial sources.
- Public datasets
- Third-party data sources
- References to external data

## Data Processing Workflow

1. Place raw data in aw/ directory
2. Create data processing scripts in src/data/
3. Generate processed datasets in processed/
4. Load processed data in notebooks or models

## File Naming Convention

Use descriptive names with dates:
- aw_sales_data_2024-01-15.csv
- processed_customer_features_2024-01-15.csv

## Data Privacy

- Never commit actual data files containing sensitive information
- Use .gitignore to exclude data files from version control
- Store sensitive files separately from the repository
