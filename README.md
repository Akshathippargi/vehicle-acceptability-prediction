

# Vehicle Acceptability Prediction

**Live Application:**  
https://vehicle-acceptability-prediction-v32ohfjxhltgctilumjnnw.streamlit.app

---

## Project Summary
This project is an **end-to-end Machine Learning application** that predicts **vehicle acceptability** (`unacc`, `acc`, `good`, `vgood`) based on pricing, safety, passenger capacity, and utility-related features.

The project demonstrates the **complete ML lifecycle**, including:
- Data preprocessing
- Feature engineering
- Model training & evaluation
- Interactive dashboard development
- Cloud deployment

---

##  Business Problem
Vehicle manufacturers, dealerships, and product teams need a reliable way to evaluate whether a vehicle configuration will be **accepted by customers** based on multiple categorical attributes.

This project solves the problem using a **supervised classification approach**.

---

##  Dataset
- **Source:** UCI Machine Learning Repository – Car Evaluation Dataset
- **Records:** 1,728
- **Features:**
  - Buying Price
  - Maintenance Cost
  - Number of Doors
  - Passenger Capacity
  - Luggage Boot Size
  - Safety Level
- **Target Variable:** Vehicle Acceptability Class

---

##  Machine Learning Methodology
- Categorical feature encoding using **One-Hot Encoding**
- Stratified train-test split
- **Random Forest Classifier** for non-linear feature interactions
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

**Final Model Accuracy:** ~**89%**

---

## Key Insights
- Safety is the strongest driver of vehicle acceptability
- High safety can offset higher buying price
- Minority classes (`good`, `vgood`) show lower recall due to class imbalance
- Probability-based predictions improve model interpretability and decision-making

---

## Application Features
- Interactive feature selection UI
- Real-time predictions
- Class probability bar chart
- Model confidence scores
- Automatically generated business insights
- Cloud-hosted Streamlit application

---

## Tech Stack
- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Web App:** Streamlit
- **Version Control:** Git & GitHub
- **Deployment:** Streamlit Cloud

---

## Project Structure
```text
vehicle-acceptability-prediction/
├── app/
│   └── app.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   └── random_forest_model.pkl
├── data/
│   └── raw/
├── notebooks/
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```
##  Run Locally

Follow the steps below to run the application on your local machine:

```bash
git clone https://github.com/Akshathippargi/vehicle-acceptability-prediction.git
cd vehicle-acceptability-prediction
pip install -r requirements.txt
python -m streamlit run app/app.py
