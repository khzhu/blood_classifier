from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from classifier.config import FIGURES_DIR, PROCESSED_DATA_DIR, PKL_PATH, MODEL_FULL_NAME
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

def _get_thresholds(y_pred_proba):
    # Only consider unique predicted probabilities
    unique_probas = np.unique(y_pred_proba)
    # Sort unique probabilities in descending order
    unique_probas_sorted = np.sort(unique_probas)[::-1]

    thresholds = np.insert(unique_probas_sorted, 0, 1.1)
    # Append 0 to the end of the thresholds
    thresholds = np.append(thresholds, 0)
    return thresholds

def _get_fpr(y_true, y_pred_proba, threshold):
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return fp / (fp + tn)

def _get_tpr(y_true, y_pred_proba, threshold):
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tp / (tp + fn)

def _plot_roc(fpr, tpr, auc, thresholds, model_name, output_path):
    # Area under curve - Gradient Boosting & Random Forest & Logistic Regression
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Model ROC Curve
    ax.plot(fpr,
            tpr,
            label=f'{model_name} (area = {auc:.2f})',
            color='blue',
            lw=3)

    # Threshold annotations
    label_kwargs = {}
    label_kwargs['bbox'] = dict(
        boxstyle='round, pad=0.3', color='lightgray', alpha=0.6
    )
    eps = 0.02  # offset
    for i in range(0, len(fpr) -1 ):
        threshold = str(np.round(thresholds[i], 2))
        ax.annotate(threshold, (fpr[i], tpr[i] - eps), fontsize=12, color='purple', **label_kwargs)

    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0]);
    ax.set_ylim([0.0, 1.05]);
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title(f'{model_name} Blood Classifier ROC Curve', fontsize=20)
    ax.legend(loc="lower right", fontsize=15)
    plt.savefig(output_path)

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "features_top50.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    model: ModelName = typer.Option(ModelName.random_forest),
    target_name: str = typer.Option("Status", "--target_name"),
):
    logger.info("Generating plot from data...")

    try:
        with open(PKL_PATH[model], 'rb') as f:
            plot_model = pickle.load(f)
        model_full_name = MODEL_FULL_NAME[model]

    except FileNotFoundError:
        print("Model is not implemented.")

    with tqdm(total=1, desc="Loading selected features...") as pbar:
        beta_df = pd.read_csv(input_path, index_col=0)
        pbar.update(1)

    X, y = beta_df.drop(target_name, axis=1), beta_df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=0)
    plot_model.fit(X_train, y_train);

    # Calculate AUC/ROC on the test set
    y_proba = plot_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    thresholds = _get_thresholds(y_proba)
    fpr = []
    tpr = []
    for threshold in thresholds:
        # FPR for the model at each of its thresholds
        fpr.append(_get_fpr(y_test, y_proba, threshold))
        # TPR for the model at each of its thresholds
        tpr.append(_get_tpr(y_test, y_proba, threshold))

    _plot_roc(fpr, tpr, auc, thresholds, model_full_name, output_path)
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
