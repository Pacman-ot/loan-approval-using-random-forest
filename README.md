# Intelligent Loan Approval Prediction System

A machine learning-powered loan approval prediction system using **Random Forest** that helps banks make fast and consistent lending decisions.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for predicting loan approval decisions based on applicant details. It includes:

- **Model Training**: A comprehensive Jupyter notebook that builds and tunes a Random Forest classifier
- **Interactive Web Interface**: A Streamlit-based application for real-time loan approval predictions
- **Production-Ready Pipeline**: Scikit-learn pipeline with preprocessing and model serialization

## 🎯 User Story

As a bank manager, I want a system that analyzes customer details like age, income, credit score, loan amount, and loan term, so that I can quickly and accurately decide whether to approve or reject a loan, reducing risk and manual effort.

## 📊 Features

### Input Features
- **Age**: Applicant's age (18-100 years)
- **Income**: Annual income
- **Credit Score**: Credit score (300-850)
- **Loan Amount**: Requested loan amount
- **Loan Term**: Loan duration in months (12-84)
- **Employment Years**: Years of employment experience
- **Existing Debt**: Current debt obligations
- **Employment Type**: Employment status (salaried, self-employed, contract, unemployed)

### Model Performance
- **Accuracy**: ~85%+
- **Precision & Recall**: Optimized using grid search with cross-validation
- **ROC-AUC**: Strong discrimination between approved and rejected loans
- **F1-Score**: Balanced performance across classes

## 📁 Project Structure

```
.
├── loan_approval_random_forest.ipynb    # Model training notebook
├── app.py                               # Streamlit web application
├── loan_approval_rf_pipeline.joblib     # Trained model (serialized)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Interactive Web App (Recommended)
Run the Streamlit application:
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501` and use the interface to input customer details and get predictions.

#### Option 2: Jupyter Notebook
Open the notebook to explore the training pipeline:
```bash
jupyter notebook loan_approval_random_forest.ipynb
```

## 📖 Notebook Sections

1. **Import Libraries & Load Data**: Sets up dependencies and loads/generates synthetic training data
2. **Data Inspection & Cleaning**: Handles missing values, outliers, and data validation
3. **Feature Encoding**: Preprocesses categorical and numerical features
4. **Train-Test Split**: Stratified split maintaining class distribution
5. **Model Training**: Trains baseline Random Forest classifier
6. **Model Evaluation**: Assesses performance with multiple metrics
7. **New Predictions**: Function to predict on new customer applications
8. **Hyperparameter Tuning**: Grid search with cross-validation for optimization
9. **Feature Importance**: Interprets model decisions through feature weights
10. **Model Serialization**: Saves and reloads the trained model

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Machine Learning | scikit-learn |
| Data Processing | pandas, numpy |
| Web Framework | Streamlit |
| Model Serialization | joblib |
| Data Visualization | matplotlib, seaborn |
| Cross-Validation | scikit-learn |

## 📊 Dataset

The project uses synthetic loan data with 2000+ samples. Features are designed to reflect real-world lending scenarios:
- Features are generated with realistic distributions
- Target variable (loan_approved) is derived from a synthetic risk scoring formula
- Balanced class distribution to avoid bias

### Data Generation Logic
The approval probability is calculated based on:
- **Positive factors**: Higher income, employment years, better credit score
- **Negative factors**: Larger loan amounts, existing debt, longer terms
- **Risk adjustments**: Unemployment status significantly increases risk

## 🎛️ Model Configuration

### Default Parameters
- **n_estimators**: 300 trees
- **max_depth**: 12
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **class_weight**: Balanced
- **random_state**: 42 (for reproducibility)

### Hyperparameter Tuning
The notebook includes grid search over:
- Number of estimators: [200, 400]
- Max depth: [None, 10, 20]
- Min samples split: [2, 5, 10]
- Min samples leaf: [1, 2, 4]

## 💾 Model Serialization

The trained model pipeline is serialized using `joblib` and saved as:
```
loan_approval_rf_pipeline.joblib
```

### Loading the Model
```python
import joblib
model = joblib.load('loan_approval_rf_pipeline.joblib')
prediction = model.predict(new_data)
```

## 🔍 Feature Importance

The top influential features for loan decisions include:
- Loan amount
- Existing debt
- Credit score
- Employment years
- Income

View the complete feature importance rankings in the notebook's visualization section.

## 📈 Making Predictions

### Via Streamlit App
1. Open the web interface
2. Enter applicant details using sliders and dropdowns
3. Adjust the approval threshold if needed
4. Click "Predict Loan Decision" to see results

### Programmatically
```python
import joblib
import pandas as pd

model = joblib.load('loan_approval_rf_pipeline.joblib')

customer = {
    'age': 35,
    'income': 78000,
    'credit_score': 720,
    'loan_amount': 150000,
    'loan_term': 60,
    'employment_years': 8,
    'existing_debt': 12000,
    'employment_type': 'salaried'
}

pred_class = model.predict(pd.DataFrame([customer]))[0]
pred_prob = model.predict_proba(pd.DataFrame([customer]))[0, 1]

print(f"Decision: {'Approved' if pred_class == 1 else 'Rejected'}")
print(f"Confidence: {pred_prob:.2%}")
```

## 🛠️ Development

### Training a New Model
If you modify the notebook and want to retrain:
1. Update hyperparameters or data if needed
2. Run all cells in the notebook
3. The trained model will be saved as `loan_approval_rf_pipeline.joblib`

### Extending the Model
- Add new features to improve predictions
- Experiment with different algorithms (XGBoost, LightGBM)
- Implement additional data validation rules
- Add monitoring and retraining pipelines

## ⚠️ Limitations & Considerations

- **Synthetic Data**: Current model is trained on synthetic data; real-world performance may vary
- **Feature Completeness**: All 8 features are required for predictions
- **Bias & Fairness**: Ensure fair lending practices; monitor for demographic disparities
- **Regulatory Compliance**: Validate against local lending regulations and fair lending laws
- **Model Drift**: Monitor model performance over time and retrain periodically

## 📝 License

This project is provided as-is for educational and business use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Real-world dataset integration
- Additional feature engineering
- Advanced model interpretability (SHAP, LIME)
- API development for enterprise integration
- Unit tests and CI/CD pipeline
- Docker containerization

## 📞 Support

For issues or questions, please open an issue in the repository or contact the project maintainer.

---

**Last Updated**: April 2026  
**Model Version**: 1.0  
**Python Version**: 3.8+
