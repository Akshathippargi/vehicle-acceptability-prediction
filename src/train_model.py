import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data
from feature_engineering import encode_features

df = load_data("data/raw/car.data")

X = df.drop("class", axis=1)
y = df["class"]

X_encoded = encode_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest_model.pkl")
print("✅ Model trained and saved")
