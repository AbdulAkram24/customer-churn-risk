# Customer Churn Risk

A small machine-learning project that explores customer churn data and trains a logistic regression classifier to estimate churn risk.

## Project Structure

```text
customer-churn-risk/
├── data/
│   └── dataset_01_customer_churn_risk.xlsx
├── src/
│   └── churn_risk.py
├── notebooks/
│   └── churn_risk_analysis.ipynb
├── .vscode/
│   ├── extensions.json                    # recommended VS Code extensions
│   ├── settings.json                      # Excel and Python workspace settings
│   └── tasks.json                         # run analysis or open the workbook
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Analysis

Run the script from the repository root:

```bash
python src/churn_risk.py
```

Or open `notebooks/churn_risk_analysis.ipynb` in Jupyter or VS Code and run the cells.

The dataset is expected to contain a `target` column and numeric predictor columns. The script reads the `Sheet1` worksheet, prints evaluation metrics and feature coefficients, and displays a confusion matrix.

## Excel in VS Code

When this folder is opened in VS Code, install the recommended extensions when prompted. The Excel Viewer extension previews `.xlsx` files inside VS Code, while Python and Jupyter support the analysis workflow. The workspace is configured to route `.xlsx` files to the Excel Viewer instead of the raw text editor.

Use the Command Palette and run either task:

- `Tasks: Run Task` > `Excel: open dataset` opens the workbook in the system Excel application.
- `Tasks: Run Test Task` runs `src/churn_risk.py` from the repository root.

The current file ending in `conversion-error.xlsx` is a quarantined copy of the malformed source. Replace it with the original dataset before running the model.
