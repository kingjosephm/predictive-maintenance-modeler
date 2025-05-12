import logging
from time import time
from typing import Dict

import xgboost as xgb
import optuna
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import StratifiedKFold


class Hyperoptimizer():

    def __init__(self, cfg, base_params, train_idx, y_upper_bound, y_lower_bound, X, target):
        self.time_identifier = cfg.data.time_identifier
        self.target_feature = cfg.data.target_feature
        self.n_trials = cfg.training_config.n_trials
        self.base_params = base_params
        self.train_idx = train_idx
        self.y_upper_bound = y_upper_bound
        self.y_lower_bound = y_lower_bound
        self.X = X
        self.target = target
        self.seed = cfg.seed


    def hyperoptimize(self) -> Dict:
        """Runs the hyperparameter optimization process using Optuna.
        The function creates a study and optimizes the objective function defined in the class. The best parameters
        are returned as a dictionary. The optimization is performed using the TPE sampler, and the number of trials
        is defined in the configuration.

        Returns:
            Dict: base parameters combined with the best trial parameters
        """

        current = time()
        logging.info("Starting hyperparameter optimization...")
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        logging.info("Hyperparameter optimization completed.")

        logging.info("Total model hyperoptimization time: %.2f seconds. \n", round(time() - current, 2))
        logging.info("Number of finished trials: %d", len(study.trials))
        logging.info("Best trial number %d", study.best_trial.number)
        logging.info("Best parameters: %s", study.best_trial.params)

        # Combine default parameters with best trial parameters
        params = {**self.base_params, **study.best_trial.params}

        return params


    def objective(self, trial: optuna.Trial) -> float:
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

        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "aft_loss_distribution": trial.suggest_categorical("aft_loss_distribution", ["normal", "logistic", "extreme"]),
            "aft_loss_distribution_scale": trial.suggest_float("aft_loss_distribution_scale", 0.1, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
        }
        params = {**self.base_params, **trial_params}

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
        X_train_cv = self.X.iloc[self.train_idx]
        y_lower_train_cv = self.y_lower_bound.iloc[self.train_idx]
        y_upper_train_cv = self.y_upper_bound.iloc[self.train_idx]
        target_train_cv = self.target.iloc[self.train_idx]

        unit_last = target_train_cv[target_train_cv[self.time_identifier] ==
                                    target_train_cv.groupby(target_train_cv.index)[self.time_identifier].transform('max')]
        stratify = unit_last[self.target_feature]

        # Define the number of folds for CV:
        sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

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
            yvalid = target_train_cv.iloc[fold_valid_idx].set_index(self.target_feature).to_records()
            ytrain = target_train_cv.iloc[fold_train_idx].set_index(self.target_feature).to_records()

            tau = target_train_cv[self.time_identifier].max() # upper time limit such that the probability of being censored is non-zero for `t > tau`

            # Compute the IPCW concordance index, negating predictions
            c_index_fold = concordance_index_ipcw(
                survival_train=ytrain,
                survival_test=yvalid,
                estimate=-pred_valid_cv,  # negate predictions, note - change as needed
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