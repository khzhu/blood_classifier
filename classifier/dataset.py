from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from pandas.api.types import is_string_dtype
from classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def _encode_categorical_variables(df):
    for col_name, col_data in df.items():
        if is_string_dtype(col_data.dtype):
            # Use factorize to get codes and unique values
            codes, uniques = pd.factorize(df[col_name])
            # Add the codes as a new column to the DataFrame
            df[col_name] = codes
    return df

@app.command()
def main(
    input_beta_path: Path = RAW_DATA_DIR / "all_betas.csv",
    input_sample_path: Path = RAW_DATA_DIR / "sample_sheet.csv",
    output_path: Path = PROCESSED_DATA_DIR / "all_features.csv"
):
    logger.info("Data Ingestion...")
    with tqdm(total=2, desc="Processing datasets...") as pbar:
        # 1. Data Ingestion : The initial step where raw data is collected from CSV files ----
        beta_data = pd.read_csv(input_beta_path, index_col=0)
        all_beta = beta_data.transpose()
        sort_beta = all_beta.sort_index()

        covar_df = pd.read_csv(input_sample_path)
        try:
            covar_df = covar_df.drop(columns=['RD_Number', 'Batch', 'Barcode', 'Array', 'Slide'])
        except KeyError:
            pass
        covar_df.set_index('Sample', inplace=True)
        sort_covar = covar_df.sort_index()
        pbar.update(1)
        # 2. Data Preprocessing: Raw data is cleaned, handled for missing values,
        # and encoded into a format suitable for machine learning algorithms. This involves:
        # - Handling missing data.
        # - Normalization or scaling.
        # - Encoding categorical variables.
        norm_beta = sort_beta.div(sort_beta.mean(axis=1), axis=0)
        sort_covar = _encode_categorical_variables(sort_covar)
        beta_df = norm_beta.join(sort_covar)
        all_betas = beta_df.transpose()
        all_betas.to_csv(output_path, index=True)
        pbar.update(2)

    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
