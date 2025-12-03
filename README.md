## Classification of Breast Cancer

This repository contains materials for a simple breast cancer classification project using the CSV dataset included in the repository (`data (2).csv`). The goal is to explore, preprocess, train, and evaluate machine learning models that predict breast cancer diagnosis (benign vs malignant).

**Status:** Active development — includes Jupyter notebook with EDA and visualizations, dataset, README and `requirements.txt`.

## Contents

- `data (2).csv` — raw dataset used for training and evaluation (placed at repository root).
- `Classification_of_Breast_Cancer.ipynb` — Jupyter notebook with exploratory data analysis, visualizations, and modeling.
- `requirements.txt` — core Python dependencies for development and model training.
- `create_ppt.py` — Python script to generate a PowerPoint presentation summarizing the project.
- `generate_assets.py` — Python script to generate visualization assets (charts, plots) for the presentation.

## Quick overview

This project demonstrates the typical workflow for a classification task:

- Inspect and clean the dataset
- Feature engineering and preprocessing
- Train one or more classifiers (for example, Logistic Regression, Random Forest, SVM)
- Evaluate using cross-validation and metrics such as accuracy, precision, recall, F1 and ROC-AUC
- Save models for reuse

## Requirements

- Python 3.8+ (recommended)
- Typical libraries used: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, altair, python-pptx

Note: `requirements.txt` contains conservative minimum versions. If you want exact pinned versions, run `pip freeze > requirements.txt` after installation.

## Setup (PowerShell)

Create and activate a virtual environment, then install packages from `requirements.txt`:

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

If you later want to freeze exact versions, run:

```powershell
pip freeze > requirements.txt
```

## How to use

1. Inspect the CSV file (`data (2).csv`) to understand column names and the target label.
2. Preprocess the data (handle missing values, encode categorical variables, scale numeric features if needed).
3. Split into train/test sets or use cross-validation.
4. Train a classifier and evaluate.

Minimal Python snippet to load the CSV and run a quick scikit-learn pipeline (example):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv(r"data (2).csv")

# Clean the data
df_cleaned = df.drop(columns=['id', 'Unnamed: 32'])

# Prepare features and target
target_col = 'diagnosis'
X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col].map({'B': 0, 'M': 1})  # Convert to numeric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print('Train score:', clf.score(X_train, y_train))
print('Test score:', clf.score(X_test, y_test))
```

Replace `target_col` with the actual label column name in `data (2).csv`.

## Quick Start

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

2. Open and run the Jupyter notebook:

```powershell
jupyter lab Classification_of_Breast_Cancer.ipynb
```

Or use VS Code's built-in notebook support to open `Classification_of_Breast_Cancer.ipynb`.

3. The notebook includes:

   - Data loading and initial exploration
   - Cleaning (dropping `id` and `Unnamed: 32` columns)
   - Diagnosis distribution analysis with interactive Altair visualizations
   - Descriptive statistics grouped by diagnosis
   - Box plots and bar charts comparing features between benign and malignant cases

4. Generate a PowerPoint presentation:

```powershell
# First, generate visualization assets
python generate_assets.py

# Then create the PowerPoint presentation
python create_ppt.py
```

This will create `Breast_Cancer_Classification_Presentation.pptx` with slides covering project overview, dataset analysis, visualizations, and key findings.

## Dataset guidance

- The dataset contains a `diagnosis` column with values 'B' (Benign) and 'M' (Malignant).
- Drop the `id` and `Unnamed: 32` columns before training (as shown in the notebook).
- The target is encoded as strings ('B'/'M') — convert to integers before training: `df['diagnosis'] = df['diagnosis'].map({'B':0,'M':1})`.
- Features include measurements with suffixes `_mean`, `_se` (standard error), and `_worst` for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Example training & model saving

Here is a minimal pattern to train and save a model (adapt `target_col`):

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv(r"data (2).csv")
df_cleaned = df.drop(columns=['id', 'Unnamed: 32'])

target_col = 'diagnosis'
X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col].map({'B': 0, 'M': 1})  # B=Benign=0, M=Malignant=1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print('Test score:', clf.score(X_test, y_test))
joblib.dump(clf, 'model.joblib')
```

## Model evaluation recommendations

- Use `classification_report` and `confusion_matrix` from `sklearn.metrics`.
- Prefer recall (sensitivity) for screening tasks; consider optimizing for high recall while monitoring precision and false positives.
- Use stratified CV (`StratifiedKFold`) if classes are imbalanced.

## Reproducibility & saving environment

- Save your trained model with `joblib.dump(model, 'model.joblib')`.
- Capture exact package versions with `pip freeze > requirements.txt` when ready to freeze environment.

## License (recommended)

I recommend adding a `LICENSE` file (MIT is common for small projects). If you want, I can add an `MIT` license now.

## What can be added next

- Complete the modeling section in the notebook (train classifiers, evaluate with confusion matrix and classification report).
- Add a ready-to-run `train.py` script that accepts `--data`, `--target`, and `--out` arguments.
- Add cross-validation and hyperparameter tuning examples.
- Add unit tests for data loading and training pipeline.
- Create a model comparison section (Logistic Regression, Random Forest, SVM, XGBoost).

Contributions are welcome! Open an issue or submit a pull request.
