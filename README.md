# Feature Engineering with Scikit-learn Transformers

A hands-on exploration of **mathematical feature transformations** — `FunctionTransformer` and `PowerTransformer` — used to fix skewed numerical data and improve machine learning model performance.

This repo contains two Jupyter notebooks, each applying a different transformation technique to a real-world dataset, with before/after visual comparisons and model accuracy benchmarking.

---

## 📌 Project Overview

Many real-world features (like `Fare` in the Titanic dataset or material quantities in the Concrete Strength dataset) are **not normally distributed** — they're skewed. Skewed data can hurt the performance of models that assume normality (like Linear/Logistic Regression).

This project demonstrates:
- How to detect skewness visually (histograms + Q-Q plots)
- How to apply `FunctionTransformer` (log transform) and `PowerTransformer` (Box-Cox)
- How these transformations affect model accuracy and R² score

---

## 📁 Notebooks

### 1. `Function_Transformer.ipynb`
**Dataset:** Titanic (`Age`, `Fare`, `Survived`)

- Cleans missing values in `Age`
- Applies `FunctionTransformer(np.log1p)` on `Fare` using `ColumnTransformer`
- Visualizes `Fare` distribution before vs. after log transform (histogram + Q-Q plot)
- Trains **Logistic Regression** and **Decision Tree** classifiers on raw vs. transformed data
- Compares accuracy scores and validates using cross-validation

### 2. `Power_Transformer.ipynb`
**Dataset:** Concrete Compressive Strength (`concrete_data.csv`)

- Visualizes distributions of all 8 input features
- Applies `PowerTransformer(method='box-cox')` to normalize skewed features
- Extracts and displays Box-Cox **lambda values** per column
- Trains a **Linear Regression** model and evaluates using **R² score**
- Validates results with cross-validation

---

## 🔑 Key Concepts Covered

- Identifying skewed distributions visually
- `FunctionTransformer` for custom transformations (e.g., log transform)
- `PowerTransformer` and the Box-Cox method
- Using `ColumnTransformer` for selective, column-specific transformations
- Measuring the real impact of preprocessing on model performance (accuracy / R²)
- Cross-validation for reliable performance estimates

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Statistics | scipy.stats |
| Machine Learning | scikit-learn |

---

## 📂 Project Structure

```
├── Function_Transformer.ipynb    # Log transform on Titanic dataset
├── Power_Transformer.ipynb       # Box-Cox transform on Concrete dataset
└── README.md
```

> **Note:** You'll need `Titanic-Dataset.csv` and `concrete_data.csv` in the project directory to run the notebooks. These are publicly available datasets (Titanic from Kaggle, Concrete Compressive Strength from UCI ML Repository).

---

## ⚙️ Setup & Usage

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
   ```

3. Launch Jupyter and run the notebooks
   ```bash
   jupyter notebook
   ```

---

## 📊 Results Summary

| Notebook | Model | Metric | Before Transform | After Transform |
|---|---|---|---|---|
| Function Transformer | Logistic Regression | Accuracy | — | ~0.68 |
| Function Transformer | Decision Tree | Accuracy | — | ~0.62 |
| Power Transformer | Linear Regression | R² | — | ~0.81 (test split) / ~0.67 (CV) |

*(Fill in your "before" numbers if you'd like a direct side-by-side comparison.)*

---

## 🚀 Future Improvements

- Add `QuantileTransformer` and `StandardScaler` comparisons
- Automate transformation selection based on skewness thresholds
- Package as a reusable preprocessing pipeline

---

## 📄 License

This project is open-sourced for learning purposes. Feel free to use or modify it.
