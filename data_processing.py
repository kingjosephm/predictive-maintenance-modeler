from typing import Tuple
import logging
from time import time
import warnings
import zipfile
import json
import io

import joblib
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from scipy.stats import entropy
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import set_seeds, is_bool_or_binary
set_seeds(42)

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class DataProcessor:

    def __init__(self, cfg: DictConfig):

        # Apply for both training and prediction modes
        self.data_path = cfg.data.data_path
        self.unit_identifier = cfg.data.unit_identifier
        self.time_identifier = cfg.data.time_identifier
        self.target_feature = cfg.data.target_feature
        assert cfg.data.sampling_n >= 1, "Sampling n must be greater than or equal to 1."
        self.sampling_n = cfg.data.sampling_n
        assert cfg.data.lag_length < cfg.data.sampling_n, "Error! `lag_length` must be less than `sampling_n`!"
        self.lag_length = min(cfg.data.lag_length, 5) # maximum 5 lags, since higher number increases chance of overfitting
        self.seed = cfg.seed

        # Apply only for training mode
        self.test_size = cfg.training_config.test_size

        # Apply only for prediction mode
        self.model_path = cfg.predict_config.model_path

    def transform(self) -> pd.DataFrame:
        """Transforms prediction dataset for model inference. This includes:
            1) loading the model and data artifacts from the zip file
            2) loading the data from the CSV file
            3) restricting to features in the previously-trained model dataset
            4) setting categorical features [if present]
            5) downsampling panel data [optional]
            6) adding feature lags [optional]
            7) normalizing numeric features using previously fitted scaler

        Returns:
            pd.DataFrame: transformed prediction dataset
        """

        current = time()
        logging.info("Transforming data for prediction..")

        # Obtain model and data artifacts from zip file
        with zipfile.ZipFile(self.model_path, "r") as zipf:

            with zipf.open("minmax_scaler.joblib") as f:
                scaler = joblib.load(f)

            with zipf.open('feature_list.json') as f:
                feature_list = json.load(f)

            with zipf.open('config.yaml') as f:
                yaml_bytes = f.read()
                model_config = OmegaConf.load(io.StringIO(yaml_bytes.decode("utf-8")))

        # Overwrite init params with loaded model config, in case they differ
        # Note: self.data_path is presumed to be correct from init
        self.unit_identifier = model_config.data.unit_identifier
        self.time_identifier = model_config.data.time_identifier
        self.target_feature = model_config.data.target_feature
        self.sampling_n = model_config.data.sampling_n
        self.lag_length = model_config.data.lag_length
        self.seed = model_config.seed

        df = pd.read_csv(self.data_path)  # Load data from CSV file

        # Note: `self.target_feature`=True should mean the outcome is observed, =False is censored
        assert df[self.target_feature].apply(is_bool_or_binary).all(), "Error! `self.target_feature` must be boolean or binary (0, 1)!"
        df[self.target_feature] = df[self.target_feature].astype(int)


        # Restrict to features in the previously-trained model
        # Note: this assumes degenerate and highly correlated features have already been dropped
        df = df[feature_list]

        # Cast categorical features to 'category' dtype
        reserved_cols = [self.unit_identifier, self.time_identifier, self.target_feature]
        assert set(reserved_cols).issubset(df.columns), "Error! `self.unit_identifier`, `self.time_identifier`, and `self.target_feature` must be in the dataframe!"
        categorical_cols = [i for i in df.columns if df[i].dtype in ['object', 'category'] and i not in reserved_cols]
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Verify `self.df` is in fact a panel dataset
        max_group_obs = df.groupby(self.unit_identifier)[self.time_identifier].count().max()
        if max_group_obs == 1 & self.sampling_n > 1:
            logging.warning("`self.df` does not appear to be a panel dataset, even though `self.sampling_n` > 1. Setting `self.sampling_n` to 1..")
            self.sampling_n = 1
            self.lag_length = 0  # no lags for cross-sectional data

        # Sort data by unit and time, in case it's not already
        df = df.sort_values([self.unit_identifier, self.time_identifier]).reset_index(drop=True)

        # Downsample panel
        if self.sampling_n > 1:
            logging.info("Downsampling panel to min(%d, len(x)) observations per '%s'..", self.sampling_n, self.unit_identifier)
            df = self.sample_group_observations(df)
        else:  # keep last observation per unit
            logging.info("Keeping only last observation per '%s'. This will not affect purely cross-sectional data.", self.unit_identifier)
            df = df[df[self.time_identifier] == df.groupby(self.unit_identifier)[self.time_identifier].transform('max')]

        df = df.set_index(self.unit_identifier)  # "hide" unit identifier for now

        # Add feature lags [optional]
        if self.lag_length > 0:
            logging.info("Adding lags up to %d periods..", self.lag_length)
            df = self.feature_lags(df)

        df = df.reset_index()  # bring back unit identifier

        # Normalization of numeric features using previously fitted scaler
        cols = [i for i in df.columns if i not in reserved_cols+categorical_cols]
        df.loc[:, cols] = scaler.transform(df.loc[:, cols])

        logging.info("Data transformation complete, took: %.2f seconds. \n", round(time() - current, 2))

        return df

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Index, pd.Index, pd.Index, MinMaxScaler]:
        """
        Initial preprocessing of data including:
            1) dropping degenerate features [if present]
            2) dropping highly correlated features [if present]
            3) setting categorical features [if present]
            4) downsampling panel data [optional]
            5) adding feature lags [optional]
            6) stratified group split for train, test, and validation sets
            7) normalizing numeric features

        :return: Tuple of
            pd.DataFrame - processed dataframe, including training, val, and test observations
            pd.Index - training set index positions
            pd.Index - val set index positions
            pd.Index - test set index positions
            MinMaxScaler - scaler object for normalizing numeric features, which can be used to transform new data
        """
        current = time()
        logging.info("Preprocessing data..")

        df = pd.read_csv(self.data_path)  # Load data from CSV file

        # Reserve columns - we don't want to drop these
        reserved_cols = [self.unit_identifier, self.time_identifier, self.target_feature]
        assert set(reserved_cols).issubset(df.columns), "Error! `self.unit_identifier`, `self.time_identifier`, and `self.target_feature` must be in the dataframe!"

        # Note: `self.target_feature`=True should mean the outcome is observed, =False is censored
        assert df[self.target_feature].apply(is_bool_or_binary).all(), "Error! `self.target_feature` must be boolean or binary (0, 1)!"
        df[self.target_feature] = df[self.target_feature].astype(int)

        # Drop any degenerate (invariant or nearly invariant) features
        degenerate_features = []
        categorical_cols = [i for i in df.columns if df[i].dtype in ['object', 'category'] and i not in reserved_cols]
        numeric_cols = [i for i in df.columns if df[i].dtype in [int, float, bool] and i not in reserved_cols]

        entropies = df[categorical_cols].apply(self.column_entropy)
        low_entropy_cols = entropies[entropies < 0.5]  # Adjust threshold as needed
        degenerate_features.extend(low_entropy_cols.index.tolist())

        cv = df[numeric_cols].std() / (df[numeric_cols].mean().replace(0, 1e-6))  # Avoid divide-by-zero
        low_cv_cols = cv[cv < 0.01]  # Adjust threshold as needed
        degenerate_features.extend(low_cv_cols.index.tolist())

        if degenerate_features:
            logging.info("Dropping degenerate features: %s", degenerate_features)
            df = df.drop(columns=degenerate_features)

        # Set categorical features
        categorical_cols = [i for i in df.columns if df[i].dtype in ['object', 'category'] and i not in reserved_cols]  # in case some dropped above
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Drop highly correlated features
        df = self.drop_highly_correlated_numeric_features(df)
        df = self.drop_highly_correlated_categorical_features(df)
        df = self.drop_highly_correlated_categorical_on_numeric_features(df)

        # Verify `self.df` is in fact a panel dataset
        max_group_obs = df.groupby(self.unit_identifier)[self.time_identifier].count().max()
        if max_group_obs == 1 & self.sampling_n > 1:
            logging.warning("`self.df` does not appear to be a panel dataset, even though `self.sampling_n` > 1. Setting `self.sampling_n` to 1..")
            self.sampling_n = 1
            self.lag_length = 0  # no lags for cross-sectional data

        # Sort data by unit and time, in case it's not already
        df = df.sort_values([self.unit_identifier, self.time_identifier]).reset_index(drop=True)

        # Downsample panel
        if self.sampling_n > 1:
            logging.info("Downsampling panel to min(%d, len(x)) observations per '%s'..", self.sampling_n, self.unit_identifier)
            df = self.sample_group_observations(df)
        else:  # keep last observation per unit
            logging.info("Keeping only last observation per '%s'. This will not affect purely cross-sectional data.", self.unit_identifier)
            df = df[df[self.time_identifier] == df.groupby(self.unit_identifier)[self.time_identifier].transform('max')]

        df = df.set_index(self.unit_identifier)  # "hide" unit identifier for now

        # Add feature lags [optional]
        if self.lag_length > 0:
            logging.info("Adding lags up to %d periods..", self.lag_length)
            df = self.feature_lags(df)

        df = df.reset_index()  # bring back unit identifier

        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)  # shuffle

        # Stratified group split for train, test, and validation sets
        train_idx, val_idx, test_idx = self.stratified_group_split(df)

        # Normalization of numeric features, not necessary but can help with convergence
        scaler = MinMaxScaler(clip=True)
        cols = [i for i in df.columns if i not in reserved_cols+categorical_cols]
        df.loc[train_idx, cols] = scaler.fit_transform(df.loc[train_idx, cols])
        df.loc[test_idx, cols] = scaler.transform(df.loc[test_idx, cols])
        df.loc[val_idx, cols] = scaler.transform(df.loc[val_idx, cols])

        logging.info("Data preprocessing complete, took: %.2f seconds. \n", round(time() - current, 2))

        return df, train_idx, val_idx, test_idx, scaler

    def column_entropy(self, series: pd.Series) -> float:
        """Calculates the entropy of a given (categorical) column in the dataframe. Entropy is a measure of uncertainty or
        unpredictability in the data. A higher entropy value indicates more disorder or randomness in the data. Features
        with low entropy (<0.5) are considered degenerate

        Args:
            series (pd.Series): input series (column) to calculate entropy for
        Returns:
            float: entropy value of the series
        """
        counts = series.value_counts(normalize=True)
        return entropy(counts, base=2)

    def drop_highly_correlated_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops highly correlated numeric features from `df` based on Pearson's correlation coefficient using a
        threshold of 0.999 to define "highly correlated". Drops n-1 features from each correlated group, where n is
        the number of correlated features.

        Args:
            df (pd.DataFrame): input dataframe, possibly with highly correlated features

        Returns:
            pd.DataFrame: modified dataframe with highly correlated features dropped
        """

        # Identify and drop highly correlated numeric features (Pearson's r > 0.999)
        threshold = 0.999
        reserved_cols = [self.unit_identifier, self.time_identifier, self.target_feature]
        numeric_cols = [i for i in df.select_dtypes(include=[int, float, bool]).columns.tolist() if i not in reserved_cols]
        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Iterate over columns and find correlated feature groups to drop. Keeps 1 feature per group, dropping others
        to_drop = set()
        for col in upper_triangle.columns:
            correlated_features = upper_triangle.index[upper_triangle[col] > threshold].tolist()
            if correlated_features:  # If there are correlated features
                correlated_features.append(col)  # Include the current column
                correlated_features = list(set(correlated_features) - to_drop)  # Remove already dropped features
                if len(correlated_features) > 1:
                    to_drop.update(correlated_features[1:])  # Keep 1 feature, drop others
        if to_drop:
            logging.info("Dropping highly correlated numeric features: %s", to_drop)
            df = df.drop(columns=to_drop)

        return df

    def drop_highly_correlated_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops highly correlated (i.e. Cramer's V > 0.999) categorical features from `df`. This is done by computing
        Cramer's V statistic for measuring association between two categorical variables, and dropping all but one
        feature from each correlated group.

        Args:
            df (pd.DataFrame): input dataframe, possibly with highly correlated features

        Returns:
            pd.DataFrame: modified dataframe with highly correlated features dropped
        """

        def cramers_v(x, y):
            """
            Compute Cramér's V statistic for measuring association between two categorical variables.

            Cramér's V is a measure of association between two nominal (categorical) variables,
            giving a value between 0 (no association) and 1 (perfect association). It is based
            on the chi-square statistic and is useful for identifying relationships between categorical features.

            Parameters:
            ----------
            x : pd.Series
                First categorical variable (must be a Pandas Series).
            y : pd.Series
                Second categorical variable (must be a Pandas Series).

            Returns:
            -------
            float
                Cramér's V value, ranging from 0 to 1:
                - 0: No association
                - 1: Perfect association

            Notes:
            ------
            - If one of the variables has only a single unique value, the function returns 0.
            - The calculation is adjusted for bias correction to prevent overestimation in small datasets.
            """
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            k_corr = k - ((k - 1) ** 2) / (n - 1)
            r_corr = r - ((r - 1) ** 2) / (n - 1)
            return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1))) if min(k_corr, r_corr) > 1 else 0

        # Identify and drop highly correlated categorical features (Cramer's V > 0.999)
        reserved_cols = [self.unit_identifier, self.time_identifier, self.target_feature]
        categorical_cols = [i for i in df.select_dtypes(include=['object', 'category']).columns if i not in reserved_cols]  # Select categorical columns
        cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)

        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2:
                    cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
                else:
                    cramers_matrix.loc[col1, col2] = 1.0

        # Set correlation threshold
        threshold = 0.999

        # Get the upper triangle of the correlation matrix (to avoid duplicate comparisons)
        upper_triangle = cramers_matrix.where(np.triu(np.ones(cramers_matrix.shape), k=1).astype(bool))

        # Initialize set of columns to drop
        to_drop = set()

        # Iterate over columns and find correlated feature groups
        for col in upper_triangle.columns:
            correlated_features = upper_triangle.index[upper_triangle[col] > threshold].tolist()
            if correlated_features:  # If correlated features exist
                correlated_features.append(col)  # Include the current column
                correlated_features = list(set(correlated_features) - to_drop)  # Remove already dropped features
                if len(correlated_features) > 1:
                    to_drop.update(correlated_features[1:])  # Keep 1 feature, drop the rest

        # Drop redundant categorical features
        if to_drop:
            logging.info("Dropping strongly related categorical features: %s", to_drop)
            df = df.drop(columns=to_drop)

        return df

    # Function to compute Mutual Information (MI) for categorical-numeric features
    def drop_highly_correlated_categorical_on_numeric_features(self, df):
        """
        Computes the correlation ratio (η²) between categorical and numeric features.
        Identifies highly correlated categorical features per numeric feature and
        drops only n-1 features per correlated group.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        df : pd.DataFrame
            DataFrame with redundant categorical features removed.
        """

        def correlation_ratio(categories, values):
            """
            Computes the correlation ratio (η²) between a categorical and numeric feature.

            Parameters:
                categories (pd.Series): Categorical variable.
                values (pd.Series): Numeric variable.

            Returns:
                float: η² value (0 to 1), where higher values indicate stronger association.
            """
            categories = categories.astype("category")
            category_groups = [values[categories == cat] for cat in categories.cat.categories]

            mean_all = values.mean()
            sst = ((values - mean_all) ** 2).sum()
            ssw = sum(((group - group.mean()) ** 2).sum() for group in category_groups)

            return 1 - ssw / sst if sst > 0 else 0

        reserved_cols = [self.unit_identifier, self.time_identifier, self.target_feature]
        categorical_cols = [i for i in df.select_dtypes(include=['object', 'category']).columns if i not in reserved_cols]
        numeric_cols = [i for i in df.select_dtypes(include=[int, float, bool]).columns.tolist() if i not in reserved_cols]

        # Compute η² matrix
        eta_matrix = pd.DataFrame(index=categorical_cols, columns=numeric_cols, dtype=float)

        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                eta_matrix.loc[cat_col, num_col] = correlation_ratio(df[cat_col], df[num_col])

        # Find categorical features that are highly correlated with numeric ones
        to_drop = set()

        # Identify correlated feature groups and drop n-1 features per group
        threshold = 0.9
        for num_col in numeric_cols:
            correlated_features = eta_matrix.index[eta_matrix[num_col] > threshold].tolist()
            correlated_features = list(set(correlated_features) - to_drop)
            if len(correlated_features) > 1:
                to_drop.update(correlated_features[1:])  # Keep one, drop the rest

        # Drop redundant features
        if to_drop:
            logging.info("Dropping categorical features with a strong numeric-categorical relationship: %s", to_drop)
            df = df.drop(columns=to_drop)

        return df


    def feature_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates over predictor columns in `df` pivoting each to respective lags, thereby converting from
        long-to-wide for only a select number of lags.

        :param df: pd.DataFrame, long-form dataset to be reshaped to wide form.
        :return: wide-form pd.DataFrame
        """
        wide_format = pd.DataFrame()

        # minimum number of observations per unit
        min_group_obs = df.groupby(self.unit_identifier)[df.columns[0]].count().min()
        if min_group_obs < self.lag_length:
            logging.warning("Warning! Minimum number of observations per '%s' is %d, which is less than `self.lag_length` of %d. Lags will be constructed up to %d periods.", self.unit_identifier, min_group_obs, self.lag_length, min_group_obs)

        # Identify invariant cols by `self.unit_identifier` - we don't want to add lags, these are fixed attributes
        uniques = df.reset_index().groupby(self.unit_identifier).apply(lambda x: x.nunique())
        invariant_cols = uniques.columns[(uniques == 1).all()].tolist()
        invariant_cols.remove(self.unit_identifier)

        cols = [i for i in df.columns if i not in [self.time_identifier, self.target_feature]+invariant_cols]
        for col in cols:
            concat_list = []
            for i in reversed(range(1, min(min_group_obs+1, self.lag_length+1))):
                concat_list.append(df.groupby(self.unit_identifier)[col].shift(i).rename(f'{col}_l{i}'))

            wide_col = pd.concat(concat_list, axis=1)
            wide_col = pd.concat([wide_col, df[col].rename(f'{col}_l0')], axis=1)  # current period
            wide_col = wide_col.bfill(
                axis=1)  # backfill any lags, in case df.groupby(self.unit_identifier)[col].shift(i).rename(f'{col}_l{i}') < lag_length

            wide_format = pd.concat([wide_format, wide_col], axis=1)  # concat with other columns

        # Concat back invariant cols
        wide_format = pd.concat([df[[self.time_identifier, self.target_feature]], wide_format, df[invariant_cols]], axis=1)

        return wide_format

    def sample_group_observations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified sampling of observations for a given unit (`self.unit_identifier`) via quantiles of time. This
        is preferred over a simple random sample per unit, which may not be representative of the full time spectrum
        that the unit contains observations for. This function calculates the quantiles of `self.time_identifier`,
        where the number of quantiles (bins) is determined by min(min(20, self.sampling_n), len(x)). The maximum number
        of observations per `unit_identifier` is 5 so that frequently-occurring `unit_identifiers`s don't dominate the
        loss.

        Args:
            df : dataframe of all units and their timepoints, and all other columns

        Returns:
            pd.DataFrame: sampled dataframe with all units, sampled
        """

        def sample_group(g: pd.DataFrame) -> pd.DataFrame:
            """Samples a group of observations based on the quantiles of the time identifier.

            Args:
                g : pd.DataFrame, group of observations for a given unit, including all timepoints

            Returns:
                pd.DataFrame : pd.DataFrame, sampled group of observations
            """
            n = len(g)
            if n <= self.sampling_n:
                return g
            # Always include the first and last observations
            first = g.iloc[[0]]
            last = g.iloc[[-1]]
            extra_needed = self.sampling_n - 2
            if extra_needed == 0:
                return pd.concat([first, last]).sort_values(self.time_identifier)
            middle = g.iloc[1:-1]
            if extra_needed > 0 and not middle.empty:
                # Determine the number of bins for quantile-based sampling
                nbins = min(extra_needed, len(middle))
                # Create quantile bins based on the time identifier
                bins = pd.qcut(middle[self.time_identifier], q=nbins, duplicates='drop')
                # For each bin, sample one observation randomly
                extra = middle.groupby(bins, observed=True).apply(lambda x: x.sample(n=1, random_state=self.seed))
                # Remove the extra group-level index added by groupby.apply
                extra = extra.reset_index(drop=True)
            else:
                extra = pd.DataFrame(columns=g.columns)
            # Concatenate the first, extra, and last observations, sorted by time
            return pd.concat([first, extra, last]).sort_values(self.time_identifier)

        min_obs_group = df.groupby([self.unit_identifier])[self.time_identifier].count().min()
        if min_obs_group < self.sampling_n:
            logging.warning("Warning! Minimum number of observations per '%s' is %d, which is less than `self.sampling_n` of %d. This will yield an unbalanced panel. Either reduce the number of `self.sampling_n` to %d or verify that an unbalanced panel is acceptable.", self.unit_identifier, min_obs_group, self.sampling_n, min_obs_group)

        sampled_rows = df[[self.unit_identifier,
                           self.time_identifier]].groupby(self.unit_identifier, group_keys=False, observed=True)\
                                                                                    .apply(sample_group).reset_index(drop=True)

        return df.merge(sampled_rows, on=[self.unit_identifier, self.time_identifier], how='inner').reset_index(drop=True)

    def stratified_group_split(self, df: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:
        """
        Creates a stratified, grouped cross-valiation on `self.unit_identifier`. Input `df` must be preprocessed and
        either a panel or cross-sectional dataset. For panel data, the last observation per `self.unit_identifier` is
        used for stratification on the target feature. Train and test sets are then merged back up with the panel
        dataset. For cross-sectional data, the target feature is stratified on the entire dataset.

        :param df: pd.DataFrame to be split
        :return: Tuple of
            pd.Index - index positions of training observations
            pd.Index - index positions of validation observations
            pd.Index - index positions of test observations
        """

        # Get last observation per unit
        undup = df[df[self.time_identifier] ==
                   df.groupby(self.unit_identifier)[self.time_identifier].transform('max')]\
                       [[self.unit_identifier, self.target_feature]].sort_values(by=[self.unit_identifier]).reset_index(drop=True)

        # Create train-test split
        x_train, x_test, _, _ = train_test_split(undup[[self.unit_identifier]], undup[self.target_feature],
                                                 test_size=self.test_size, random_state=self.seed,
                                                 stratify=undup[self.target_feature])

        # Filter out test observations
        undup = undup[~undup[self.unit_identifier].isin(x_test[self.unit_identifier])]

        # Create train-validation split
        x_train, x_val, _, _ = train_test_split(undup[[self.unit_identifier]], undup[self.target_feature],
                                                 test_size=0.2, random_state=self.seed,
                                                 stratify=undup[self.target_feature])

        assert set(x_train[self.unit_identifier]).intersection(x_test[self.unit_identifier]).intersection(x_val[self.unit_identifier]) == set(), "Error! Train, val, and test sets are not mutually exclusive!"

        assert set(x_train[self.unit_identifier]).union(x_test[self.unit_identifier]).union(x_val[self.unit_identifier]) == set(df[self.unit_identifier]), "Error! Train, val, and test sets do not cover all observations!"

        # Specify index positions for train, val, and test sets
        train_idx = df[df[self.unit_identifier].isin(x_train[self.unit_identifier].tolist())].index
        val_idx = df[df[self.unit_identifier].isin(x_val[self.unit_identifier].tolist())].index
        test_idx = df[df[self.unit_identifier].isin(x_test[self.unit_identifier].tolist())].index

        assert set(train_idx).intersection(val_idx).intersection(test_idx) == set(), "Error! Train, val, and test set indices are not mutually exclusive!"

        return train_idx, val_idx, test_idx