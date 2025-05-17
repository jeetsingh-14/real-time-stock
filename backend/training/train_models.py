import pandas as pd
import numpy as np
import logging
import joblib
import os
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from collections import Counter
from stockmarket_analysis.backend.utils.config import MYSQL_CONNECTION_STRING

# Configure logging
log_dir = os.path.join(Path(__file__).resolve().parents[2], "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_models.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def connect_to_database():
    try:
        logger.info("Connecting to MySQL database")
        engine = create_engine(MYSQL_CONNECTION_STRING)
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def load_training_data(engine):
    try:
        logger.info("Loading training data from merged_price_sentiment table")
        query = "SELECT * FROM merged_price_sentiment"
        df = pd.read_sql(query, engine)

        logger.info(f"Successfully loaded {len(df)} records for training")

        if 'sentiment' not in df.columns or 'timestamp' not in df.columns or 'price_now' not in df.columns:
            logger.error("Missing required columns in dataset")
            raise ValueError("Required columns missing")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        # Compute target
        df = df.sort_values(['symbol', 'timestamp'])
        df['future_price'] = df.groupby('symbol')['price_now'].shift(-3)
        df['price_change'] = df['future_price'] - df['price_now']

        def classify(change):
            if change > 0.5:
                return "UP"
            elif change < -0.5:
                return "DOWN"
            else:
                return "NEUTRAL"

        df['target'] = df['price_change'].apply(classify)
        df = df.dropna(subset=['target'])

        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def prepare_data(df):
    try:
        logger.info("Preparing features and target")

        X = df[['sentiment', 'day_of_week', 'hour', 'month']]
        y = df['target']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        logger.info(f"Class distribution: {dict(Counter(y_encoded))}")

        stratify = y_encoded if min(Counter(y_encoded).values()) >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.4, random_state=42, stratify=stratify
        )

        logger.info(f"Prepared {X_train.shape[0]} training and {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test, label_encoder
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    try:
        logger.info("Training and evaluating models")

        class_weights = 'balanced'
        xgb_weight = y_train.shape[0] / (3 * np.bincount(y_train) + 1e-6)
        xgb_weight_dict = {i: w for i, w in enumerate(xgb_weight)}

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1)
        }

        results = {}
        all_classes = label_encoder.classes_
        all_classes = np.array([c for c in all_classes if c is not None])

        class_mapping = {
            0: "DOWN",
            1: "NEUTRAL",
            2: "UP",
            3: "STRONG_UP"
        }

        target_names = np.array([class_mapping.get(label_encoder.transform([c])[0], f"Class_{c}") for c in all_classes])
        unique_labels = label_encoder.transform(all_classes)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test,
                y_pred,
                labels=unique_labels,
                target_names=target_names,
                zero_division=0
            )

            results[name] = {
                'model': model,
                'accuracy': acc,
                'report': report
            }

            logger.info(f"{name} - Accuracy: {acc:.4f}")
            logger.info(f"{name} - Report:\n{report}")

        return results
    except Exception as e:
        logger.error(f"Error training and evaluating models: {str(e)}")
        raise

def save_best_model(results, label_encoder):
    try:
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']

        logger.info(f"Best model: {best_model_name}")

        model_package = {
            'model': best_model,
            'label_encoder': label_encoder
        }

        # Use correct path: backend/model/
        model_dir = os.path.join(Path(__file__).resolve().parents[1], "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pkl")

        joblib.dump(model_package, model_path)
        logger.info(f"Best model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving best model: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting model training process")

        engine = connect_to_database()
        df = load_training_data(engine)

        X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)

        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder)

        save_best_model(results, label_encoder)

        logger.info("Training process completed successfully")

    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
