# Customer Churn Risk Prediction

A machine learning project to predict customer churn risk using Logistic Regression.

## Overview

This project builds a predictive model to identify customers at risk of churning. It includes comprehensive preprocessing, model training, evaluation metrics, and feature importance analysis.

## Features

- **Data Preprocessing**: Data cleaning and feature engineering
- **Model**: Logistic Regression for binary classification
- **Train/Test Split**: Proper data partitioning for model evaluation
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Score
- **Visualizations**: Confusion matrix and feature importance plots
- **Feature Analysis**: Coefficient interpretation for model explainability

## Project Structure

```
customer-churn-risk/
├── src/                    # Source code
│   └── churn_risk.py      # Main model implementation
├── data/                   # Datasets
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Model outputs and visualizations
└── requirements.txt        # Python dependencies
```

## Requirements

See `requirements.txt` for dependencies. Install with:

```bash
pip install -r requirements.txt
```

## Usage

Run the main model:

```bash
python src/churn_risk.py
```

## Results

Model performance and visualizations are saved to the `results/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Abdul Akram
