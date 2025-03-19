from typing import Tuple, Optional, Dict
import logging
from time import time
import warnings
warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost

from utils import set_seeds, is_bool_or_binary
set_seeds(42)

warnings.filterwarnings("ignore", category=DeprecationWarning)  #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class DataProcessor:

    def __init__(self,
                 unit_identifier: str,
                 time_identifier: str,
                 target_feature: str,
                 categorical_features_prefix: str,
                 test_size: float = 0.1,
                 seed: int = 42,
                 lag_length: int = 2,
                 sampling_n: int = 5,
                 oversample: bool = False
                 ):

        self.unit_identifier = unit_identifier
        self.time_identifier = time_identifier
        self.target_feature = target_feature
        self.categorical_features_prefix = categorical_features_prefix
        self.test_size = test_size
        self.seed = seed
        self.sampling_n = max(1, sampling_n)  # minimum 1, negative values not possible
        self.oversample = oversample
        assert lag_length < sampling_n, "Error! `lag_length` must be less than `sampling_n`!"
        self.lag_length = min(lag_length, 5) # maximum 5 lags, since higher number increases chance of overfitting


    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial preprocessing of data including:
            1) change target to bool
            2) Downsample the panel to reduce total observations and the influence of outliers. Done with stratified random sampling technique, where `self.time_identifier` used to stratify observations per `self.unit_identifier`
            3) addition of lagged features for each `self.unit_identifier`
            4) creation of final outcome features for xgboost interval censoring
        :param df: pd.DataFrame, input dataframe
        :return: pd.DataFrame
        """
        current = time()
        logging.info("Preprocessing data..")

        # Recode label to bool -> whether event observed or censored. Needed for classical survival analysis
        # Note: `self.target_feature`=True should mean the outcome is observed, =False is censored
        assert df[self.target_feature].apply(is_bool_or_binary).all(), "Error! `self.target_feature` must be boolean or binary (0, 1)!"

        # Verify `self.df` is in fact a panel dataset
        max_group_obs = df.groupby(self.unit_identifier)[self.time_identifier].count().max()
        if max_group_obs == 1 & self.sampling_n > 1:
            logging.warning(f"`self.df` does not appear to be a panel dataset, even though `self.sampling_n` > 1. Setting `self.sampling_n` to 1..")
            self.sampling_n = 1
            self.lag_length = 0  # no lags for cross-sectional data

        # Sort data by unit and time, in case it's not already
        df = df.sort_values([self.unit_identifier, self.time_identifier]).reset_index(drop=True)

        # Downsample panel
        if self.sampling_n > 1:
            logging.info(f"Downsampling panel to min({self.sampling_n}, len(x)) observations per '{self.unit_identifier}'..")
            df = self.sample_group_observations(df)
        else:  # keep last observation per unit
            logging.info(f"Keeping only last observation per '{self.unit_identifier}'. This will not affect purely cross-sectional data.")
            df = df[df[self.time_identifier] == df.groupby(self.unit_identifier)[self.time_identifier].transform('max')]

        df = df.set_index(self.unit_identifier)

        # Set categorical features
        cats = [i for i in df.columns if self.categorical_features_prefix in i]
        df[cats] = df[cats].astype('category')

        # Add feature lags [optional]
        if self.lag_length > 0:
            df = self.feature_lags(df)

        logging.info(f"Data preprocessing complete, took: {round(time() - current, 2)} seconds. \n")

        return df.reset_index()


    def split_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Second data preprocessing step including:
            1) Stratified train-test-split
            2) Standardizing X matrix
        :param df: pd.DataFrame
        :return: Tuple of [pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
        """
        current = time()
        logging.info("Splitting & transforming data..")

        # Stratified train-test-val-split
        X_train, X_test, y_train, y_test = self.stratified_train_test_val_split(df)  # removes self.unit_identifier from index

        # Normalization (note - only strictly required for SVM)
        scaler = MinMaxScaler(clip=True)
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        logging.info(f"Splitting and transforming data complete, took: {round(time() - current, 2)} seconds. \n")

        return (X_train, X_test, y_train, y_test)


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
            logging.warning(f"Warning! Minimum number of observations per '{self.unit_identifier}' is {min_group_obs}, which is less than `self.lag_length` of {self.lag_length}. Lags will be constructed up to {min_group_obs} periods.")

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
        """Stratified sampling of observations for a given unit (`self.unit_identifier`) via quantiles of time. This is preferred over
        a simple random sample per unit, which may not be representative of the full time spectrum that the unit contains
        observations for. This function calculates the quantiles of `self.time_identifier`, where the number of
        quantiles (bins) is determined by min(min(20, self.sampling_n), len(x)). The maximum number of observations per `unit_identifier` is
        5 so that frequently-occurring `unit_identifiers`s don't dominate the loss.

        Args:
            df (pd.DataFrame): dataframe of all units and their timepoints, and all other columns

        Returns:
            pd.DataFrame: sampled dataframe
        """
        min_obs_group = df.groupby([self.unit_identifier])[self.time_identifier].count().min()
        if min_obs_group < self.sampling_n:
            logging.warning(f"Warning! Minimum number of observations per '{self.unit_identifier}' is {min_obs_group}, which is less than `self.sampling_n` of {self.sampling_n}. This will yield an unbalanced panel. Either reduce the number of `self.sampling_n` to {min_obs_group} or verify that an unbalanced panel is acceptable.")

        quantile = df.groupby([self.unit_identifier])[self.time_identifier].apply(lambda x:  pd.qcut(x, q=min(min(20, self.sampling_n), len(x)), labels=False)).reset_index(drop=True)

        return df.groupby([self.unit_identifier, quantile]).apply(lambda x: x.sample(1, random_state=self.seed)).reset_index(drop=True)


    def Xy_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Splits `df` into X matrix and y vectors
        :param df: pd.DataFrame, preprocessed train or test set
        :return: Tuple of (pd.DataFrame, np.ndarray)
        """
        y_cols = [self.target_feature, self.time_identifier]
        x_cols = [i for i in df.columns if i not in y_cols+[self.unit_identifier]]  # remove `self.unit_identifier` since not needed
        
        X = df.loc[:, x_cols]
        y = df.loc[:, y_cols]
        y = y.set_index(self.target_feature).to_records()  # converts to structured array, which is required for sklearn-survival
        return X, y


    def stratified_train_test_val_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Creates a stratified, grouped train-test-val split on `self.unit_identifier`. Necessary to prevent data leakage 
        across splits. Train and val splits are lumped together in train/X_train objects.
        
        :param df: pd.DataFrame
        :return: Tuple of 
            pd.DataFrame - X_train
            pd.DataFrame - X_test
            np.ndarray - y_train
            np.ndarray - y_test
        """

        # Target feature invariant across `self.unit_identifier` due to `self.preprocess()`, doesn't matter which one we select
        undup = df[[self.unit_identifier, self.target_feature]].drop_duplicates()

        # Create train-test split
        x_train, x_test, _, _ = train_test_split(undup[[self.unit_identifier]], undup[self.target_feature], test_size=self.test_size,
                                                random_state=self.seed, stratify=undup[self.target_feature])
        
        # Create train and test sets, also shuffle
        test = df[df[self.unit_identifier].isin(x_test[self.unit_identifier].tolist())].copy()\
            .sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        train = df[df[self.unit_identifier].isin(x_train[self.unit_identifier].tolist())].copy()\
            .sample(frac=1.0, random_state=self.seed).reset_index(drop=True)  # includes val observations

        # Split into X and y
        X_train, y_train = self.Xy_split(train)
        X_test, y_test = self.Xy_split(test)

        return (X_train, X_test, y_train, y_test)