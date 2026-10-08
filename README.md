# Machine Learning-Based Clinical Decision Support System for Postpartum Haemorrhage Detection

> A machine learning-based clinical decision support system that predicts a patient's risk of postpartum haemorrhage (PPH) and explains what is driving that prediction, so healthcare professionals get both a risk score and a reason behind it.

## Project Overview

Postpartum haemorrhage is one of the leading causes of maternal death worldwide, and it is especially dangerous in places where blood loss is hard to detect early. This project builds a machine learning system that predicts a patient's risk of PPH using maternal and clinical information, explains why the model made that prediction, and turns the prediction into a clear, actionable recommendation for healthcare staff.

The system combines two machine learning models into a single hybrid model, uses SHAP (an explainability technique) to show which factors drove each prediction, and wraps everything in a simple web interface built with Streamlit.

## Key Highlights

- Dataset: 30,000 simulated maternal health records
- Data source: Maternal Mortality and Postpartum Haemorrhage Dataset (Electric Sheep Africa, via Hugging Face)
- Models: Random Forest and XGBoost
- Final model: Hybrid ensemble combining Random Forest (Weighted) and XGBoost (SMOTE) using weighted soft voting, 30% RF + 70% XGBoost
- Explainability: SHAP (SHapley Additive exPlanations)
- Classification threshold: 0.40 for PPH vs non-PPH
- Risk categories: Low, Moderate, High (separate three-tier thresholds used for clinical recommendations)
- Final model performance: 93.65% accuracy, 90.97% precision, 43.45% recall, 58.81% F1-score, 75.87% ROC-AUC
- Five engineered clinical features: Obstetric Risk Score, Pregnancy Load Index, Labour Stress Index, Antenatal Care Score, Delivery Complexity Score

## Problem Statement

Postpartum haemorrhage is one of the top causes of maternal death, and the risk is highest in places with limited healthcare infrastructure, including Nigeria and much of Sub-Saharan Africa. The traditional way of detecting it relies on visually estimating blood loss and watching vital signs, but visual estimation is often wrong by 30 to 50 percent, and the body can keep vital signs looking normal until blood loss becomes severe. By the time it's obvious something is wrong, it can already be a crisis.

Existing AI models built for this problem tend to have two issues. First, many were trained on data from a single hospital or region, so they don't generalise well elsewhere. Second, most are "black box" models that give a risk score without explaining why, which makes clinicians less likely to trust or act on them. This project tries to address both: a model trained to detect PPH risk, paired with an explanation of what's driving each prediction.

## Project Objectives

- Design and train a machine learning model using clinical data, with particular attention to maternal demographics, haemoglobin levels, and the Obstetric Shock Index, to predict PPH risk.
- Incorporate an explainable AI approach (SHAP) so predictions can be understood, not just trusted blindly.
- Develop a simple, standalone, web-based interface for the system.
- Evaluate the system's performance using standard classification metrics: precision, accuracy, recall, and F1-score.

## Dataset

### Dataset Description

The project uses the Maternal Mortality and Postpartum Haemorrhage Dataset, created by Electric Sheep Africa and sourced through Hugging Face. It contains 30,000 **simulated** (not real patient) maternal health cases, combining scenarios from Basic Emergency Obstetric and Newborn Care (BEmONC), Comprehensive Emergency Obstetric and Newborn Care (CEmONC), and Community Birth settings.

The variables cover maternal demographics, pregnancy history, antenatal care, labour and delivery details, complications, and interventions. The target variable is `pph`, which indicates whether postpartum haemorrhage occurred.

<img width="975" height="285" alt="image" src="https://github.com/user-attachments/assets/4dca3822-371c-4a23-9e9b-3fd35c1f72e1" />
<p align="center">
  <b>Dataset Sample</b>
</p>

### Data Preprocessing

Preprocessing followed these steps, in order:

1. **Data quality checks and missing value handling** — the dataset was reviewed for completeness before modelling began.
2. **Data leakage removal** — several variables were removed because they were only known *after* PPH occurred or after treatment began, which would let the model "cheat" by seeing information from the future. Removed variables included estimated blood loss, PPH severity, blood transfusion status, PPH cause, uterine massage, manual placenta removal, surgical intervention, referral status, referral delay, and hospital days.
3. **Feature engineering** — five new clinical scores were built by combining related variables (explained below).
4. **Feature selection** — an Extra Trees Classifier was used to rank features by importance and reduce the dataset to the most predictive ones. *(Note: the document states two different numbers for how many features were kept — see "Notes for Esther" below.)*
5. **Categorical encoding** — variables like education, marital status, and delivery type were converted into numeric form using one-hot encoding.
6. **Train/test split** — the data was split 80/20 for training and testing, using stratified sampling so both sets kept the same proportion of PPH and non-PPH cases.

## Methodology

<img width="975" height="580" alt="image" src="https://github.com/user-attachments/assets/29b1a21d-ba99-42e6-84fb-ff3e65d9fcd6" />
<p align="center">
  <b>ML CDSS Framework</b>
</p>

<img width="975" height="564" alt="image" src="https://github.com/user-attachments/assets/1e83d5b6-7b3c-4859-b45e-991df4a85995" />
<p align="center">
  <b>ML CDSS Architecture</b>
</p>

### 1. Data Preparation

Before modelling, five engineered features were created to give the models a more clinically meaningful view of each patient, rather than relying on many raw variables independently:

- **Obstetric Risk Score** — combines maternal age, parity, and gravidity into one overall risk indicator.
- **Pregnancy Load Index** — combines multiple pregnancy status and antenatal care attendance.
- **Labour Stress Index** — combines prolonged labour and obstructed labour into one measure of labour-related complications.
- **Antenatal Care (ANC) Score** — combines number of antenatal visits with whether the patient completed at least four visits.
- **Delivery Complexity Score** — combines previous caesarean history with the current delivery method.

<img width="803" height="658" alt="image" src="https://github.com/user-attachments/assets/2dc223f8-1e60-4819-a6e5-652e8ff16323" />
<p align="center">
  <b>Feature Engineering Python Code</b>
</p>


<img width="927" height="553" alt="image" src="https://github.com/user-attachments/assets/47d204a5-6bed-4ae9-8d25-21152e14561c" />
<p align="center">
  <b>Bar Chart showing the Top 20 Features Chosen</b>
</p>


> This shows which variables, including the engineered scores, the model relied on most.

### 2. Model Development

**Random Forest** was chosen for its strong baseline accuracy and its ability to handle a mix of numeric and categorical data. It builds many decision trees on random subsets of data and features, which reduces overfitting. It was implemented with 300 trees (`n_estimators = 300`).

<img width="592" height="444" alt="image" src="https://github.com/user-attachments/assets/48d63374-f379-4f50-adb8-6566ff6177c4" />
<p align="center">
  <b>Python Code for the Random Forest Model</b>
</p>


**XGBoost** was chosen for its ability to catch harder, less obvious patterns. Unlike Random Forest, where trees are built independently, XGBoost builds trees sequentially, with each new tree correcting the mistakes of the ones before it. It was also implemented with 300 estimators.

<img width="602" height="461" alt="image" src="https://github.com/user-attachments/assets/da4d91f1-8553-40e3-a579-f421cfeea5cf" />
<p align="center">
  <b>Python Code for the XGBoost Model</b>
</p>

### 3. Model Evaluation

Each model was evaluated using five standard classification metrics: Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC. Recall mattered most for this project, because missing an at-risk patient (a false negative) is far more dangerous than incorrectly flagging a low-risk patient.

### 4. Ensemble Model

Rather than picking one model, the final system combines the predictions of both Random Forest and XGBoost using **weighted soft voting** — each model outputs a probability, and the two probabilities are combined using a weighted average.

Four different weighting combinations were tested. The configuration that used **30% Random Forest (Weighted) + 70% XGBoost (SMOTE)** was selected as the final model, because it achieved the highest recall (43.45%) among the tested combinations — meaning it caught more true PPH cases than the alternatives, even though a couple of other weightings scored marginally higher on F1-score and ROC-AUC. For a system meant to flag at-risk patients, missing fewer real cases mattered more than a slightly higher overall score.

<img width="632" height="281" alt="image" src="https://github.com/user-attachments/assets/e11c1755-08ff-4402-bd2d-4ad11c85c2d3" />
<p align="center">
  <b>Python Code for the Ensemble Model</b>
</p>

### 5. Risk Classification

The hybrid model's output is a probability between 0 and 1. Two different thresholds are used in the system:

- **Binary classification threshold (0.40):** used to decide whether a case counts as "PPH" or "Non-PPH" during model evaluation.
- **Three-tier risk categories:** used by the clinical recommendation engine to turn that probability into a practical risk level:

| Predicted Probability | Risk Level | Recommendation |
|---|---|---|
| Below 0.30 | Low Risk | Routine monitoring and standard maternal care |
| 0.30 to below 0.70 | Moderate Risk | Increase monitoring and prepare for possible intervention |
| 0.70 and above | High Risk | Activate the PPH management protocol and begin immediate clinical intervention |

## Model Evaluation and Results

The table below shows the final comparison between the individual models and the selected hybrid ensemble, taken from the project's final ensemble weight comparison.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest (Weighted) | — | — | — | — | — |
| XGBoost (SMOTE) | — | — | — | — | — |
| **Hybrid (30% RF + 70% XGBoost) — Selected** | **93.65%** | **90.97%** | **43.45%** | **58.81%** | **75.87%** |

*(Individual RF-Weighted and XGBoost-SMOTE standalone scores appear in the document's broader model comparison table alongside several other variants — see "Notes for Esther" if you'd like me to pull specific rows into this table.)*

<img width="631" height="578" alt="image" src="https://github.com/user-attachments/assets/9b328c85-d2d8-48b4-a227-da47d8e4c2f1" />
<p align="center">
  <b>ROC AUC for the Random Forest Model</b>
</p>


<img width="625" height="605" alt="image" src="https://github.com/user-attachments/assets/057573f7-9f90-452a-b47b-9c4181e08451" />
<p align="center">
  <b>ROC AUC for the XGBoost Model</b>
</p>


<img width="653" height="683" alt="image" src="https://github.com/user-attachments/assets/ae346698-fc3d-4304-b4ca-fda672ab0d4a" />
<p align="center">
  <b>ROC AUC for the Ensemble Model</b>
</p>


<img width="1063" height="423" alt="image" src="https://github.com/user-attachments/assets/38dfdad7-9b80-48a0-8d11-2f4db4caff68" />
<p align="center">
  <b>Confusion Matrix for the Random Forest, XGBoost and Ensemble Model</b>
</p>


In practical terms: on a test set of roughly 6,000 patients, the hybrid model correctly identified 272 out of 626 actual PPH cases, with 354 missed (false negatives) and 27 patients incorrectly flagged as high risk (false positives). The hybrid model caught more true PPH cases than either Random Forest or XGBoost alone (263 and 268, respectively), which is why it was selected — but the recall of 43.45% means the model still misses more than half of actual PPH cases. This is an honest limitation of the system, not a small one, and is discussed further below.

## Explainable AI

### SHAP

SHAP (SHapley Additive exPlanations) was used to explain the XGBoost (SMOTE) component of the model. It works by estimating how much each individual feature — like maternal age, haemoglobin level, or the engineered Obstetric Risk Score — pushed a specific prediction higher or lower. This was used in two ways in the project:

- **Globally**, to understand which features matter most across the whole dataset.
- **Locally**, to explain a single patient's individual prediction.

<img width="922" height="677" alt="image" src="https://github.com/user-attachments/assets/640450af-e483-42a9-8a65-7dcf039f358f" />
<p align="center">
  <b>SHAP BeeSwarm Plot showing the contribution of each features</b>
</p>


<img width="848" height="733" alt="image" src="https://github.com/user-attachments/assets/cc7dc2d9-1b45-47e1-934c-734459cd0e8c" />
<p align="center">
  <b>Bar Chart Showing how each feature contributes to the output of the model's  output</b>
</p>


<img width="975" height="580" alt="image" src="https://github.com/user-attachments/assets/ed6682fb-a90c-4b46-8876-c1f32d5b5562" />
<p align="center">
  <b>SHAP Waterfall Plot showing the contribution of each feature to the model's output</b>
</p>


The results showed that the Obstetric Risk Score was the single most influential feature overall, followed by maternal age and the Pregnancy Load Index. This is a meaningful result on its own: it suggests that the engineered features actually captured something clinically useful, not just noise.

*(Note: your document's literature review discusses LIME as an alternative explainability technique but explicitly states it was not used in the final implementation. I've left it out of this README accordingly — see "Notes for Esther.")*

## Clinical Decision Support Workflow

<img width="880" height="1192" alt="image" src="https://github.com/user-attachments/assets/8833cb82-8be0-47b3-b891-6c49349d6ee7" />
<p align="center">
  <b>ML CDSS Flow Chart</b>
</p>


## Results and Key Findings

- The hybrid model outperformed both individual models at catching true PPH cases, improving true positives from 263 (Random Forest alone) and 268 (XGBoost alone) to 272.
- The engineered clinical features, especially the Obstetric Risk Score and Pregnancy Load Index, were consistently the most influential predictors — suggesting that combining related clinical variables into composite scores added real value, rather than just adding complexity.
- Despite the improvement, recall stayed modest at 43.45%, meaning the system still misses a significant share of true PPH cases. The document is explicit about this: the system should be treated as a decision-support tool, not a replacement for clinical judgement.

<img width="1049" height="445" alt="image" src="https://github.com/user-attachments/assets/2fbfd6ea-73e4-4973-a6d2-4bd647776cfe" />
<p align="center">
  <b>ML CSS Website</b>
</p>


<img width="1045" height="473" alt="image" src="https://github.com/user-attachments/assets/4354c254-b73c-4fc4-ae92-634da6366625" />
<p align="center">
  <b>ML CSS Website</b>
</p>

## Technologies Used

- Python
- Pandas
- NumPy *(used for numerical operations; confirm inclusion if not explicitly used in your notebook)*
- Scikit-learn (Random Forest, Extra Trees Classifier, train/test split)
- XGBoost
- SHAP
- Joblib (model saving/loading)
- Matplotlib
- Streamlit (web interface)
- Hugging Face Datasets (data loading)
- Google Colab / Jupyter Notebook (development environment)

## Repository Structure

*Note: your project document doesn't describe an actual repository layout, so this is a suggested structure only — adjust it to match your real folders and files.*

```text
pph-cdss/
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── README.md
└── requirements.txt
```

## How to Run

Your project document doesn't include installation steps, package versions, or exact file names, so I can't write accurate run instructions without guessing. If you'd like this section filled in, send me your actual notebook filenames, a requirements list (or your imports), and how you run the Streamlit app, and I'll write this section properly.

## Limitations

- **The dataset is simulated, not real patient data.** This is explicitly stated in your document and it's the single most important limitation — the model has not been tested on real clinical records.
- **Recall is modest (43.45%).** The model still misses more than half of actual PPH cases in testing, which matters a great deal for a life-threatening condition.
- **No real-world clinical validation yet.** The document states that validation using real hospital data is required before any practical deployment.
- **Scope is limited to PPH only.** The system does not address other maternal complications like eclampsia or gestational diabetes.
- **Not integrated with hospital systems.** It's a standalone application, not connected to any Electronic Health Record (EHR) system.

## Future Improvements

- Validate the system using real-world clinical data from hospitals and maternal healthcare centres.
- Test additional models such as LightGBM, CatBoost, or neural networks to see if they improve on current performance.
- Integrate with Electronic Health Record (EHR) systems for automatic patient data retrieval.
- Extend the approach to other maternal health complications, such as sepsis.
- Continue using explainability techniques like SHAP in any future version, to keep predictions transparent to clinicians.

## Ethical Considerations

This system is designed to support, not replace, clinical judgement — the final decision always rests with a qualified healthcare professional. Explainability (via SHAP) was a deliberate design choice, since a risk score clinicians can't interrogate is a risk score they're unlikely to trust or act on responsibly. Because the training data is simulated rather than drawn from real patients, questions of bias and fairness across different patient populations remain untested and should be a priority before any real-world use. Any future version that uses real patient data would also need to seriously address data privacy and consent, which this project, using synthetic data, did not need to resolve.

## Disclaimer

This project is an academic research prototype developed as part of a final-year university project. It is not clinically validated, is not intended to diagnose patients, and should not be used as a standalone diagnostic system or as a substitute for professional medical judgment.

## Author

**Esther Matthew**
[LinkedIn](https://www.linkedin.com/in/esther-matthew) · [GitHub](https://github.com/Queeenest147) · [Medium](https://esther-matthew.medium.com)
