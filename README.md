# Blood Classifier
## Overview
A non-invasive classifier predicts Isocitrate DeHydrogenase mutation status in ctDNA methylation profiles 
analyzed with Illumina Infinium Epic arrays. IDH mutation status is a vital diagnostic and prognostic 
indicator in diffuse gliomas, the most prevalent malignant primary brain tumor in adults. Assessing IDH 
mutation preoperatively through peripheral blood is noninvasive and can greatly assist neurosurgical planning.

## Contents
Significant efforts have focused on developing computationally efficient statistical methods to analyze epigenetic 
changes in cancer. In this study, I examined a semi-supervised recursively partitioned mixture model (SS-RPMM, from Koestler et al.) 
and LightGBM (LGBM), an open-source gradient-boosting framework from Microsoft used for feature selection. LGBM outperformed 
SS-RPMM in accuracy and overall performance. Incorporating clinical data for dimensionality reduction not only reduces 
computational demands but also helps preserve informative loci that might otherwise be lost when the number 
of dimensions is chosen arbitrarily.

## How to set up and run the blood classifier

1. This repository should first be cloned from GitHub:
```
git clone https://github.com/khzhu/blood-classifier.git
```
2. You need to have a conda environment set up on your local machine before running the classification. 
Instructions for installing Miniconda are available [here](https://www.anaconda.com/docs/getting-started/miniconda/install). To install classifier requirements, 
use the Makefile and enter
```
make requirements
```
3. The raw input files are beta values stored in CSV format, with columns as sample names and rows as CpG loci.
Clinical phenotypes relevant for prediction, such as histology, tumor grade, and IDH status, are also stored 
in CSV format. Raw IDAT files from Illumina scanners need to be converted to beta values using tools such as 
the minfi R package, then passed through Quality Control. You may find this [tutorial](https://www.bioconductor.org/packages//release/bioc/vignettes/minfi/inst/doc/minfi.html) useful.


4. The input CSV files need to be placed in the data/raw folder before running the classifier. To execute the full workflow, 
enter 'make pipeline'. You can also execute individual steps manually by using 
'make' followed by the specific target, like 'make train'.


5. The classifier provides three models: Random Forest, Logistic Regression, and Gradient Boosting. Use the 'model' input parameter to select the one you prefer when making predictions.

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks.
├── pyproject.toml     <- Project configuration file with package metadata.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as DOC, HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── requirements.txt   <- The requirements file for reproducing the analysis environment
└── classifier   <- Source code for use in this project.
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── features.py             <- Code to create features for modeling
    ├── modeling
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    └── plots.py                <- Code to create visualizations
```
