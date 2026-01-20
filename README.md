
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red)](https://vehicle-acceptability-prediction-v32ohfjxhltgctilumjnnw.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Akshathippargi/vehicle-acceptability-prediction)

# 🚗 Vehicle Acceptability Prediction

🔗 **Live Demo:**  
https://vehicle-acceptability-prediction-v32ohfjxhltgctilumjnnw.streamlit.app

---

## 📌 Overview
This project is an end-to-end **Machine Learning application** that predicts **vehicle acceptability** (`unacc`, `acc`, `good`, `vgood`) based on pricing, safety, passenger capacity, and utility features.  
The complete ML lifecycle — data preprocessing, model training, evaluation, and deployment — is implemented and deployed on **Streamlit Cloud**.

---

## 🎯 Problem Statement
Vehicle manufacturers and dealers need a reliable way to assess whether a vehicle configuration will be acceptable to customers based on multiple categorical factors.  
This project solves the problem using a **Random Forest classification model**.

---

## 📂 Dataset
- **Source:** UCI Machine Learning Repository – Car Evaluation Dataset
- **Total Records:** 1,728
- **Features:**
  - Buying Price
  - Maintenance Cost
  - Number of Doors
  - Passenger Capacity
  - Luggage Boot Size
  - Safety Level
- **Target Variable:** Vehicle Acceptability Class

---

## 🧠 Machine Learning Approach
- One-hot encoding for categorical features
- Stratified train-test split
- **Random Forest Classifier**
- Evaluation using accuracy, precision, recall, and F1-score

**Final Model Accuracy:** ~**89%**

---

## 📊 Key Insights
- Safety is the strongest determinant of vehicle acceptability
- High safety can compensate for higher price
- Minority classes (`good`, `vgood`) have lower recall due to class imbalance
- Probability-based predictions improve interpretability

---

## 🖥️ Streamlit Application Features
- Interactive user interface
- Real-time acceptability prediction
- Class probability bar chart
- Model confidence scores
- Automatically generated business insights

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Git & GitHub
- Streamlit Cloud

---

## 📁 Project Structure

```text
vehicle-acceptability-prediction/
├── app/
│   └── app.py                  # Streamlit application
│
├── src/
│   ├── data_preprocessing.py   # Data loading & preprocessing
│   ├── feature_engineering.py  # Feature encoding & transformations
│   ├── train_model.py          # Model training script
│   └── evaluate_model.py       # Model evaluation & metrics
│
├── models/
│   └── random_forest_model.pkl # Trained ML model
│
├── data/
│   └── raw/
│       ├── car.data
│       ├── car.names
│       └── car.c45-names
│
├── notebooks/                  # Jupyter notebooks (optional)
├── reports/                    # Evaluation reports & visuals
│
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

