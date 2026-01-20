import joblib
from data_preprocessing import load_data
from feature_engineering import encode_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = load_data("data/raw/car.data")

X = encode_features(df.drop("class", axis=1))
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("models/random_forest_model.pkl")
preds = model.predict(X_test)

print(classification_report(y_test, preds))
