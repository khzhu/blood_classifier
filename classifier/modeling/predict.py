from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import typer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, r2_score
from classifier.config import MODELS_DIR, PROCESSED_DATA_DIR, PKL_PATH, MODEL_FULL_NAME
import warnings
from sklearn import set_config
from enum import Enum

warnings.filterwarnings('ignore')
set_config(transform_output="pandas")

class ModelName(str, Enum):
    random_forest = "rf"
    logreg = "logreg"
    xgboost = "xgboost"

app = typer.Typer()

def _validation(model, model_name, X_test, y_test):
    model_full_name = MODEL_FULL_NAME[model_name]

    # Calculate AUC/ROC on the test set
    y_proba = model.predict_proba(X_test)[:, 1]
    # Calculate accuracy
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_full_name} Accuracy: {accuracy:.4f}")

    auc = roc_auc_score(y_test, y_proba)
    print(f'{model_full_name} AUC on test set:{auc:.2f}')

    r2 = r2_score(y_test, y_pred)
    print(f"{model_full_name} R-squared score: {r2:.2f}")

    loss = log_loss(y_test, y_proba)
    print(f"{model_full_name} Log-loss:, {loss: .4f}")

    return y_pred, y_proba

def _predict(X, y, model_name, predictions_path):
    with open(PKL_PATH[model_name], 'rb') as f:
        model = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=0)
    # Fit the model on the train data
    model.fit(X_train, y_train)
    y_pred, y_proba = _validation(model, model_name, X_test, y_test)

    res_df = X_test.assign(true_target=y_test, predicted_target=y_pred)
    # Rearrange the dataframe to exclude unnecessary columns
    res_df = res_df.loc[:, ~res_df.columns.str.startswith('cg')]
    numeric_cols = res_df.select_dtypes(include=[np.number]).columns
    res_df[numeric_cols] = res_df[numeric_cols].astype('Int64')
    res_df['true_target'] = res_df['true_target'].map({0: 'mut', 1: 'wt'})
    res_df['predicted_target'] = res_df['predicted_target'].map({0: 'mut', 1: 'wt'})
    # Save the dataframe in a CSV file
    res_df = res_df.reset_index(names=['Sample'])
    res_df.to_csv(predictions_path, index=False)

@app.command()
def main(
        model: ModelName = typer.Option(ModelName.random_forest),
        target_name: str = typer.Option("Status", "--target_name"),
        feature_path: Path = PROCESSED_DATA_DIR / "features-top50.csv",
        predictions_path: Path = PROCESSED_DATA_DIR / "test-predictions.csv"
    ):

    logger.info(f"Performing inference for {model}...")

    with tqdm(total=2, desc="Loading all features...") as pbar:
        beta_df = pd.read_csv(feature_path, index_col=0)
        X, y = beta_df.drop(target_name, axis=1), beta_df[target_name]
        pbar.update(1)

        try:
            _predict(X, y, model, predictions_path)
        except FileNotFoundError:
            print("Model is not implemented.")
        pbar.update(2)

    logger.success("Inference complete.")

if __name__ == "__main__":
    app()
