from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = Path(__file__).parent / "data" / "landslide_training_data.csv"

NUMERIC = ["rainfall_1h_mm", "rainfall_24h_mm", "soil_moisture_pct",
           "groundwater_level_m", "slope_angle_deg", "vegetation_cover_pct",
           "human_disturbance"]
CATEGORICAL = ["soil_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "landslide_event"

def load_dataset():
    return pd.read_csv(DATA_PATH)

def train_model(data):
    X, y = data[FEATURES], data[TARGET]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=180, max_depth=12,
                                              class_weight="balanced", random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, pred),
               "roc_auc": roc_auc_score(y_test, scores)}
    # Build readable feature names from the fitted transformer.
    cat_encoder = model.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["onehot"]
    feature_names = NUMERIC + list(cat_encoder.get_feature_names_out(CATEGORICAL))
    importances = model.named_steps["classifier"].feature_importances_
    # Pipeline classifier importances correspond to transformed feature columns.
    class ModelWrapper:
        def __init__(self, pipeline, names, importances):
            self.pipeline = pipeline
            self.feature_names = names
            self.feature_importances_ = importances
        def predict_proba(self, X):
            return self.pipeline.predict_proba(X)
    return ModelWrapper(model, feature_names, importances), metrics, feature_names

def predict_risk(model, values):
    X = pd.DataFrame([values])
    return float(model.predict_proba(X)[0, 1])

def risk_label(probability):
    if probability < 0.30:
        return ("Low", "green", "Keep monitoring conditions. Low estimated risk does not mean zero risk.")
    if probability < 0.60:
        return ("Moderate", "orange", "Increase observation frequency and review local rainfall and slope conditions.")
    if probability < 0.80:
        return ("High", "red", "Promptly review conditions with local experts and follow official guidance.")
    return ("Very High", "red", "Treat as a high model score requiring expert review. Follow official emergency instructions.")
