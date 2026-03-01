from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from classifier.config import MODELS_DIR, PROCESSED_DATA_DIR
import warnings
from sklearn import set_config

warnings.filterwarnings('ignore')
set_config(transform_output="pandas")

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features-top50.csv",
    target_name: str = typer.Option("Status", "--target_name"),
    pipeline_path: Path = MODELS_DIR / "lgbm-pipeline.pkl"
):
    logger.info("Training LGBM...")

    with tqdm(total=1, desc="10 fold cross validation and calculate accuracy...") as pbar:
        beta_df = pd.read_csv(features_path, index_col=0)
        X, y = beta_df.drop(target_name, axis=1), beta_df[target_name]

        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=0)

        # 10-fold CV accuracy
        scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring="accuracy")
        print("CV Accuracy:", scores.mean())
        pbar.update(1)

    logger.success("Modeling training complete.")

if __name__ == "__main__":
    app()
