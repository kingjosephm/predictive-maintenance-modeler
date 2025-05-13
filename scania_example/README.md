# Application of Predictive Maintenance Modeler to Scania Dataset

Files in this directory illustrate how to use Predictive Maintenance Modeler on an example multivariate time series dataset.

# Data
Data for this example are publicly-available multivariate time-series data from Scania trucks, which was specifically created for time-to-event analysis. [This whitepaper](https://arxiv.org/pdf/2401.15199) explains the background on the dataset, which can be downloaded [here](https://stockholmuniversity.app.box.com/s/anmg5k93pux5p6decqzzwokp9vuzmdkh).

[This notebook](data_merging.ipynb) merges these raw data into one dataset and carries out some initial preprocessing steps, including:
- creating the `target_feature`
- using groupby forward-fill for missing values
- to backward fill missing values, I use [MICE imputation](https://medium.com/@kunalshrm175/multivariate-imputation-by-chained-equations-mice-2d3efb063434) using [Bayesian ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html).

This preprocessed dataset can be downloaded here:

- [Training data](https://drive.google.com/file/d/1LT_95MKmR57WAtpoC9VDZrmH_1kL6A1a/view?usp=sharing)
- [Test data](https://drive.google.com/file/d/1IchHJmjbKQDkpLVGWfpx-PIoHr3NhUGu/view?usp=sharing)

Exploratory data analysis of the preprocessed dataset is done in [this notebook](exploratory_data_analysis.ipynb).