from pathlib import Path
import pickle
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from classifier.config import PROCESSED_DATA_DIR, MODELS_DIR
import warnings
from sklearn import set_config

warnings.filterwarnings('ignore')
set_config(transform_output="pandas")
app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "all_features.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_top50.csv",
    target_name: str = typer.Option("Status", "--target_name"),
    pipeline_path: Path = MODELS_DIR / "lgbm_pipeline.pkl"
):
    logger.info("Feature Engineering...")
    with tqdm(total=3, desc="Feature Selection...") as pbar:
        # 3. Feature Engineering/Selection:
        # selecting the most relevant existing features to enhance model performance.
        beta_df = pd.read_csv(input_path, index_col=0)
        all_betas = beta_df.transpose()

        data = all_betas.drop(target_name, axis=1)
        target = all_betas[target_name]
        pbar.update(1)

        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)

        pipeline.fit(data, target)
        pbar.update(2)
        # Step 1: variance threshold mask
        var_mask = pipeline.named_steps["var_thresh"].get_support()

        # Step 2: selector mask (applied after variance filtering)
        select_mask = pipeline.named_steps["lgbm_select"].get_support()

        # First reduce columns by variance threshold
        remaining_features = data.columns[var_mask]

        # Then apply top-50 selection
        top50_features = remaining_features[select_mask]
        features_top50 = pd.concat([all_betas[list(top50_features)], all_betas[target_name]], axis=1)
        features_top50.to_csv(output_path, index=True)
        pbar.update(3)

    logger.success("Features generation complete.")

if __name__ == "__main__":
    app()
