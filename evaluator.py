import os
import logging
from typing import Dict, Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm, logistic
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score

matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
plt.rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.5


class Evaluator:
    def __init__(self, output_path: str, params: Dict, eval_results: Dict, pred_train: np.ndarray,
                 pred_test: np.ndarray, ytrain: pd.DataFrame, ytest: pd.DataFrame) -> None:
        self.output_path = output_path
        self.params = params
        self.eval_results = eval_results
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.ytrain = ytrain
        self.ytest = ytest

        # Convert to format required by concordance_index_ipcw
        self.survival_train = self.ytrain.set_index(self.ytrain.columns[-1]).to_records()  # left-most column is time
        self.survival_test = self.ytest.set_index(self.ytest.columns[-1]).to_records()

        self.metrics = {}

        os.makedirs(self.output_path, exist_ok=True)


    def run(self) -> None:
        """Runs the evaluation process, including plotting losses, calculating C-index, and plotting survival curves."""

        self.plot_losses()
        self.calculate_cindex()
        time_grid, surv_probs = self.plot_survival_curves()

        # Calculate integrated brier score
        self.metrics['Integrated Brier Score'] = integrated_brier_score(self.survival_train, self.survival_test,
                                                                        surv_probs, time_grid)

        logging.info("Integrated Brier Score: %.4f", self.metrics['Integrated Brier Score'])

        # Save performance metrics as JSON
        with open(os.path.join(self.output_path, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f)

    def plot_losses(self) -> None:
        """Plots the training and validation losses by iteration. Losses are measured as the negative log-likelihood.
        Due to scaling, the y-axis is in log10 scale."""

        # Access the loss results
        train_losses = self.eval_results['train']['aft-nloglik']
        val_losses = self.eval_results['valid']['aft-nloglik']

        # Convert to DataFrame for convenience
        loss_df = pd.DataFrame({
            'train_loss': train_losses,
            'validation_loss': val_losses
        })
        loss_df.index.name = 'iteration'
        loss_df.index += 1

        # Plot the losses
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.plot(loss_df.index, loss_df['train_loss'], label='Train')
        plt.plot(loss_df.index, loss_df['validation_loss'], label='Validation')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel(r'$\log_{10}(\text{Negative Log-Likelihood})$')
        plt.title('Training and Validation Loss by Iteration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'losses.png'), dpi=200)
        plt.close()


    def calculate_cindex(self) -> None:
        """Calculates the IPCW concordance index (C-index) for the training and test sets."""

        self.metrics['Concordance Index'] = {}

        self.metrics['Concordance Index']['train'] = concordance_index_ipcw(survival_train=self.survival_train,
                                                                            survival_test=self.survival_train,
                                                                            estimate=-self.pred_train,  # negate predictions, note - change as needed
                                                                            tau=self.ytrain.iloc[:, 0].max())[0]

        self.metrics['Concordance Index']['test'] = concordance_index_ipcw(survival_train=self.survival_train,
                                                                           survival_test=self.survival_test,
                                                                           estimate=-self.pred_test,  # negate predictions, note - change as needed
                                                                           tau=self.ytest.iloc[:, 0].max())[0]

        logging.info("Concordance Index (train): %.4f", self.metrics['Concordance Index']['train'])
        logging.info("Concordance Index (test): %.4f", self.metrics['Concordance Index']['test'])


    def plot_survival_curves(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot survival curves for all test observations.
        """

        test_min = self.ytest.iloc[:, 0].min() + 1e-5 # avoid zero
        test_max = self.ytest.iloc[:, 0].max()
        time_grid = np.linspace(test_min, test_max, 500, endpoint=False)

        surv_probs = self.__survival_curve(time_grid=time_grid,
                                           predicted_medians=self.pred_test,
                                           sigma=self.params['aft_loss_distribution_scale'],
                                           distribution=self.params['aft_loss_distribution'])


        # Unduplicate target, in case of panel data, which is critical for Kaplan-Meier
        max_time = self.ytest.groupby(self.ytest.index)[self.ytest.columns[0]].transform('max')
        undup_target = self.ytest[self.ytest[self.ytest.columns[0]] == max_time]

        # Plot the survival curves
        plt.clf()
        plt.figure(figsize=(10, 6))
        kmf = KaplanMeierFitter(label='Observed')
        kmf.fit(durations=undup_target.iloc[:, 0], event_observed=undup_target.iloc[:, 1])
        kmf.survival_function_.plot()
        plt.plot(time_grid, np.mean(surv_probs, axis=0), label='Forecasted', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Probability of Survival')
        plt.title('Forecasted & Observed Survival using Holdout Set')
        plt.legend()
        plt.savefig(os.path.join(self.output_path, 'survival_curves.png'), dpi=200)
        plt.close()

        return time_grid, surv_probs

    def __survival_curve(self, time_grid, predicted_medians, sigma=1.0, distribution='normal'):
        """
        Compute the survival curves for all observations given a time grid.

        Parameters
        ----------
        time_grid : np.array
            1D array of time points at which to evaluate the survival function.
        predicted_medians : np.array
            1D array (n_obs,) of predicted median survival times from XGBoost.
        sigma : float, optional
            Scale parameter (aft_loss_distribution_scale), default is 1.0.
        distribution : str
            Distribution type, one of 'normal', 'logistic', or 'extreme'.
            Default is 'normal'.

        Returns
        -------
        surv_probs : np.array
            A 2D array of shape (n_obs, n_timepoints) where each row is the survival curve
            for one observation.
        """
        assert distribution in ['normal', 'logistic', 'extreme'], f"Unsupported distribution: {distribution}"

        if distribution =='extreme':
            mu = np.log(predicted_medians) + sigma * 0.3662041  # 0.3662041 approximates -ln(ln2)
            mu = mu[:, None]  # Reshape mu for broadcasting: (n_obs, 1)
        else:
            # Convert predicted medians to the location parameter mu on the log scale:
            mu = np.log(predicted_medians)[:, None]  # Shape (n_obs, 1)

        # Get the log of the time grid; shape (1, n_timepoints)
        log_time_grid = np.log(time_grid)[None, :]

        # Compute the survival probability at each time point for every observation:
        if distribution == 'normal':
            surv_probs = 1 - norm.cdf((log_time_grid - mu) / sigma)
        elif distribution == 'logistic':
            surv_probs = 1 - logistic.cdf((np.log(log_time_grid) - mu) / sigma)
        else:
            surv_probs = np.exp(-np.exp((log_time_grid - mu) / sigma))
        return surv_probs