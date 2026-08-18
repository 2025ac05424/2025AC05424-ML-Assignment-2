# Machine Learning Assignment 2 — Purchase Intent Model Lab

**Student:** Richa Tripathi  
**BITS ID:** 2025AC05424  
**Course:** Machine Learning — BITS Pilani WILP M.Tech (AIML/DSE)

## 1. Problem Statement

The objective of this assignment is to implement and compare five classification algorithms on the same classification dataset and the same held-out test data. The models predict whether an online shopping session results in a purchase. Performance is compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). A Streamlit application is included to upload test data, select a model, and view evaluation results.

The positive class is **Revenue = 1**, meaning that the shopping session ended in a purchase.

## 2. Dataset Description

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository (dataset ID 468, DOI: `10.24432/C5F88Q`). The raw dataset contains **12,330 sessions and 17 input features**. The target column is `Revenue`.

UCI reports 10,422 non-purchase sessions and 1,908 purchase sessions in the raw data. The source reports no missing values. The dataset is distributed under **CC BY 4.0**.

Dataset citation: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

### Data preparation used here

- Raw rows: **12,330**
- Raw predictors: **17**
- Missing target rows: **0**
- Exact duplicate complete records removed: **125**
- Cleaned usable rows: **12,205**
- Split: **80% training / 20% test**, stratified, `random_state=42`
- Training rows: **9,764**
- Held-out test rows: **2,441**
- The identical held-out test rows are used for all five models.

Preprocessing is fitted only on training data using scikit-learn pipelines. Numerical variables are median-imputed; numerical scaling is used for Logistic Regression, kNN and Gaussian Naive Bayes. Categorical variables are most-frequent imputed and one-hot encoded. Tree-based models use unscaled numerical values.

## 3. GitHub Repository Link

**GitHub Repository:** [https://github.com/2025ac05424/2025AC05424-ML-Assignment-2](https://github.com/2025ac05424/2025AC05424-ML-Assignment-2)

## 4. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The complete modelling, preprocessing, evaluation, and model-saving code is contained in `2025AC05424_ML_Assignment2.ipynb`. The fitted model pipelines used by Streamlit are stored in the `model/` folder as `.joblib` files.

## 5. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8558 | 0.7376 | 0.5373 | 0.5654 | 0.5510 | 0.4654 |
| kNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Gaussian Naive Bayes | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Random Forest (Ensemble) | 0.9066 | 0.9256 | 0.7770 | 0.5654 | 0.6545 | 0.6123 |

AUC is calculated from the predicted probability of the positive class `Revenue=1`, rather than from hard predicted labels. Precision, Recall, and F1 use the purchase class as the positive class.

## 6. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It gives strong AUC (**0.8996**) and Precision (**0.7524**), but Recall is lower (**0.4136**), so it misses a notable share of actual purchase sessions. |
| Decision Tree | Recall (**0.5654**) is better than Logistic Regression and kNN, but its AUC (**0.7376**) and MCC (**0.4654**) are lower than the strongest models. |
| kNN | Accuracy is **0.8734**, but Recall is only **0.3901**, showing that many purchase sessions are classified as non-purchases. |
| Gaussian Naive Bayes | Recall is very high (**0.9791**), but Precision is only **0.1756**. It predicts too many non-purchase sessions as purchases, producing weak Accuracy, F1, and MCC. |
| Random Forest (Ensemble) | It provides the strongest overall balance: Accuracy **0.9066**, AUC **0.9256**, Precision **0.7770**, Recall **0.5654**, F1 **0.6545**, and MCC **0.6123**. |
| Overall Winner for the dataset | **Random Forest (Ensemble)** |

## 7. Overall Winner

**Random Forest (Ensemble)** is selected as the overall winner. It has the highest MCC (**0.6123**), F1 (**0.6545**), AUC (**0.9256**), Accuracy (**0.9066**), and Precision (**0.7770**) among the five models. Gaussian Naive Bayes has the highest Recall, but its very low Precision and MCC show that its predictions are not well balanced.

## 8. Streamlit Application

The Streamlit application, **Purchase Intent Model Lab**, provides:

- CSV test-data upload
- model selection dropdown
- comparison of all five models on labeled test data
- display of Accuracy, AUC, Precision, Recall, F1 and MCC
- confusion matrix
- classification report
- bundled `test_data.csv` download
- prediction-only mode if the target column is not included

**Live Streamlit App:** [https://2025ac05424-ml-assignment-2.streamlit.app/](https://2025ac05424-ml-assignment-2.streamlit.app/)

## 9. Repository Contents

```text
.
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── 2025AC05424_ML_Assignment2.ipynb
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## 10. Local Setup and Execution

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the Jupyter Notebook top-to-bottom to reproduce the data cleaning, split, training, evaluation and saved model files.

Launch the Streamlit app from the repository root:

```bash
streamlit run app.py
```

For evaluation in the application, upload `test_data.csv`, which contains the 17 raw input features and the true `Revenue` target.

## 11. Reproducibility Notes

- `random_state=42`
- stratified 80:20 train-test split
- same held-out test rows for every model
- positive class: `Revenue=1` (purchase)
- AUC computed using positive-class probability
- all preprocessing is fitted on training data inside each saved pipeline
- no training takes place inside the Streamlit application
- no source-data download is required when the application starts
- Streamlit deployment uses a compatible Python 3.12 / scikit-learn 1.6.1 environment so the saved pipelines can be loaded consistently

## 12. Limitations

The purchase class is less frequent than the non-purchase class, so Accuracy alone can be misleading. No resampling or hyperparameter optimization is used in the primary comparison. The purpose is a transparent and reproducible academic comparison of the five required classifiers.
