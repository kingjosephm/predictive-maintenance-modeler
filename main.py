import os
import logging
from time import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sklearn.model_selection import StratifiedKFold

from utils import set_seeds, plot_losses, survival_curves, plot_survival_curves
from data_processing import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['HYDRA_FULL_ERROR'] = '1'  # better error trace


matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.5

@hydra.main(config_path="./", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a model using centralized learning

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    set_seeds(cfg.seed)

    # 1. Print parsed config
    logging.info(OmegaConf.to_yaml(cfg))

    # 2. Read and preprocess data
    df = pd.read_csv(cfg.data_path, compression='infer')

    # 3. Initialize data processor class
    dp = DataProcessor(
        unit_identifier = cfg.unit_identifier,
        time_identifier  = cfg.time_identifier,
        target_feature = cfg.target_feature,
        test_size = cfg.test_size,
        seed = cfg.seed,
        lag_length = cfg.lag_length,
        sampling_n = cfg.sampling_n
    )

    # 4. Preprocess data
    df, train_idx, val_idx, test_idx = dp.preprocess(df)


    # 5. Create labels for xgboost interval censoring
    assert cfg.sampling_n >= 1, "Sampling n must be greater than or equal to 1."
    if cfg.sampling_n == 1:  # simple cross-sectional data
        y_lower_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_lower_bound')
        y_upper_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_upper_bound')
        y_upper_bound = y_upper_bound.where(df[cfg.target_feature] == 1, +np.inf)  # where condition true, use orig value, else +inf
    else:  # panel data -> interval censored
        y_lower_bound = pd.Series(df.groupby(cfg.unit_identifier)[cfg.time_identifier].shift(1).fillna(0), name='y_lower_bound')  # missing values are left-censored and set to 0
        y_upper_bound = pd.Series(df[cfg.time_identifier].copy(), name='y_upper_bound')
        y_upper_bound = y_upper_bound.where(df[cfg.target_feature] == 1, +np.inf)

    # Separate target feature from X matrix
    df = df.set_index(cfg.unit_identifier)  # note - this has no effect on positional indexing below
    target_features = [cfg.time_identifier, cfg.target_feature]
    target = df[target_features].copy()
    target[cfg.target_feature] = target[cfg.target_feature].astype(bool)
    X = df.drop(columns=cfg.target_feature)

    # 5. Train model
    if cfg.hyperoptimize:

        def objective(trial: optuna.Trial) -> float:
            """Hyperparameter optimization objective function.
            This function is called by Optuna to evaluate the performance of a set of hyperparameters. Hyperoptimization
            is performed using stratified k-fold group cross-validation on the training set. Performance measures are
            based on the average IPCW concordance index across the folds. Trial pruning is implemented if >= 2 folds
            are unpromising, based on the current `trial.study.best_value` and a margin of 0.05.

            Args:
                trial (optuna.Trial): Current optuna trial

            Returns:
                float: mean IPCW concordance index across the folds
            """

            params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "tree_method": "hist",
                "verbosity": 0,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "aft_loss_distribution": trial.suggest_categorical("aft_loss_distribution", ["normal", "logistic", "extreme"]),
                "aft_loss_distribution_scale": trial.suggest_float("aft_loss_distribution_scale", 0.1, 10.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
            }

            cv_scores = []

            # Set up early trial pruning
            unpromising_count = 0  # number of folds (below) with unpromising results, used to prune unpromising trials
            margin = 0.05  # margin to subtract from the best score to determine if a fold is unpromising
            try:  # will yield ValueError if no previous trials
                best_overall_score = trial.study.best_value
                dynamic_threshold = best_overall_score - margin
            except ValueError:
                dynamic_threshold = None  # No threshold for the very first trial

            # Use StratifiedKFold on the training set
            X_train_cv = X.iloc[train_idx]
            y_lower_train_cv = y_lower_bound.iloc[train_idx]
            y_upper_train_cv = y_upper_bound.iloc[train_idx]
            target_train_cv = target.iloc[train_idx]

            unit_last = target_train_cv[target_train_cv[cfg.time_identifier] ==
                                        target_train_cv.groupby(target_train_cv.index)[cfg.time_identifier].transform('max')]
            stratify = unit_last[cfg.target_feature]

            # Define the number of folds for CV:
            sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

            # Iterate over unit-level folds.
            for fold_count, (fold_train_unit_idx, fold_valid_unit_idx) in enumerate(sgkf.split(unit_last, stratify)):

                # Get the unit identifiers for this fold.
                train_units = unit_last.iloc[fold_train_unit_idx].index
                valid_units = unit_last.iloc[fold_valid_unit_idx].index

                # Map these back to the panel training indices.
                fold_train_idx = np.flatnonzero(X_train_cv.index.isin(train_units))
                fold_valid_idx = np.flatnonzero(X_train_cv.index.isin(valid_units))

                # Create DMatrix for training and validation within the fold
                dtrain_cv = xgb.DMatrix(X_train_cv.iloc[fold_train_idx],
                                        label_lower_bound=y_lower_train_cv.iloc[fold_train_idx],
                                        label_upper_bound=y_upper_train_cv.iloc[fold_train_idx],
                                        enable_categorical=True)

                dvalid_cv = xgb.DMatrix(X_train_cv.iloc[fold_valid_idx],
                                        label_lower_bound=y_lower_train_cv.iloc[fold_valid_idx],
                                        label_upper_bound=y_upper_train_cv.iloc[fold_valid_idx],
                                        enable_categorical=True)

                cv_evals = {}
                bst_cv = xgb.train(
                    params,
                    dtrain_cv,
                    num_boost_round=10_000,
                    evals=[(dtrain_cv, 'train'), (dvalid_cv, 'valid')],
                    early_stopping_rounds=50,
                    evals_result=cv_evals,
                    callbacks=[],
                    verbose_eval=False
                )

                # Get predictions for the validation fold
                pred_valid_cv = bst_cv.predict(dvalid_cv)

                # Check if any predictions are np.inf, meaning the model failed to converge
                if np.any(np.isinf(pred_valid_cv)):
                    raise optuna.exceptions.TrialPruned("Infinite predictions encountered.")

                # Build a structured survival array for the validation fold as this is required for `concordance_index_ipcw`
                yvalid = target_train_cv.iloc[fold_valid_idx].set_index(cfg.target_feature).to_records()
                ytrain = target_train_cv.iloc[fold_train_idx].set_index(cfg.target_feature).to_records()

                tau = target_train_cv[cfg.time_identifier].max() # upper time limit such that the probability of being censored is non-zero for `t > tau`

                # Compute the IPCW concordance index, negating predictions
                c_index_fold = concordance_index_ipcw(
                    survival_train=ytrain,
                    survival_test=yvalid,
                    estimate=-pred_valid_cv,
                    tau=tau
                )[0]
                cv_scores.append(c_index_fold)

                # If we have a dynamic threshold, check the fold's c-index.
                if dynamic_threshold is not None and c_index_fold < dynamic_threshold:
                    unpromising_count += 1

                # Report current mean c-index (each fold gets a unique step).
                trial.report(np.mean(cv_scores), step=fold_count)
                # If at least 2 folds are unpromising, prune the trial.
                if fold_count >= 1 and unpromising_count >= 2:
                    raise optuna.exceptions.TrialPruned()

            return np.mean(cv_scores)

        current = time()
        logging.info("Starting hyperparameter optimization...")
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=True)
        logging.info("Hyperparameter optimization completed.")

        logging.info("Total model hyperoptimization time: %.2f seconds. \n", round(time() - current, 2))
        logging.info("Number of finished trials: %d", len(study.trials))
        logging.info("Best trial number %d", study.best_trial.number)
        logging.info("Best parameters: %s", study.best_trial.params)

    else:

        dtrain = xgb.DMatrix(X.iloc[train_idx],
                            label_lower_bound=y_lower_bound[train_idx],
                            label_upper_bound=y_upper_bound[train_idx],
                            enable_categorical=True)
        ytrain = target.iloc[train_idx].set_index(cfg.target_feature).to_records()

        dvalid = xgb.DMatrix(X.iloc[val_idx],
                    label_lower_bound=y_lower_bound[val_idx],
                    label_upper_bound=y_upper_bound[val_idx],
                    enable_categorical=True)

        dtest = xgb.DMatrix(X.iloc[test_idx],
            label_lower_bound=y_lower_bound[test_idx],
            label_upper_bound=y_upper_bound[test_idx],
            enable_categorical=True)
        ytest = target.iloc[test_idx].set_index(cfg.target_feature).to_records()

        # Set default parameters
        params = {
            'verbosity': 0,
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'tree_method': 'hist',
            'learning_rate': 0.05,
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 1.0,
            'max_depth': 5,
            'lambda': 0.5,
            'alpha': 0.05
        }

        # Train gradient boosted trees using AFT loss
        current = time()
        evals_result = {}
        bst = xgb.train(params,
                        dtrain,
                        num_boost_round=10_000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,
                        evals_result=evals_result)

        logging.info("Total model training time: %.2f seconds. \n", round(time() - current, 2))

        # Plot the training and validation losses
        plot_losses(evals_result, HydraConfig.get().runtime.output_dir)

        # Estimated median survival time, we treat as risk score, where higher risk scores indicate higher risk of failure
        pred_test = bst.predict(dtest)
        pred_train = bst.predict(dtrain)

        # Calculate IPCW Concordance Index
        c_indices = {}
        c_indices['train'] = concordance_index_ipcw(survival_train=ytrain, survival_test=ytrain,
                                              estimate=-pred_train, tau=target.iloc[train_idx][cfg.time_identifier].max())[0]
        c_indices['test'] = concordance_index_ipcw(survival_train=ytrain, survival_test=ytest,
                                              estimate=-pred_test, tau=target.iloc[train_idx][cfg.time_identifier].max())[0]

        logging.info("Concordance Index (train): %.4f", c_indices['train'])
        logging.info("Concordance Index (test): %.4f", c_indices['test'])


        # Plot survival curves
        test_min = target.iloc[test_idx][cfg.time_identifier].min() + 1e-5 # avoid zero
        test_max = target.iloc[test_idx][cfg.time_identifier].max()
        time_grid = np.linspace(test_min, test_max, 100, endpoint=False)
        surv_probs = survival_curves(time_grid=time_grid, predicted_medians=pred_test,
                                     sigma=params['aft_loss_distribution_scale'],
                                     distribution=params['aft_loss_distribution'])

        plot_survival_curves(surv_probs=surv_probs, time_grid=time_grid, target=target.iloc[test_idx],
                             output_path=HydraConfig.get().runtime.output_dir)

        # Calculate integrated brier score
        ibs = integrated_brier_score(ytrain, ytest, surv_probs, time_grid)

        logging.info("Integrated Brier Score: %.4f", ibs)

if __name__ == '__main__':

    main() # pylint: disable=E1120
