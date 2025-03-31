import random
import os
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic
from lifelines import KaplanMeierFitter

def set_seeds(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_bool_or_binary(value: Any) -> bool:
    """Tests if a value is a boolean or binary (0, 1).

    Args:
        value Any: a value to test

    Returns:
        bool: boolean indicating if the value is a boolean or binary
    """
    return isinstance(value, bool) or value in (0, 1)

def plot_losses(evals_result: Dict, output_path: str) -> None:
    """Plots the training and validation losses by iteration. Losses are measured as the negative log-likelihood.
    Due to scaling, the y-axis is in log10 scale.

    Args:
        evals_result (Dict): dictionary of training and validation losses by iteration
        output_path (str): HydraConfig.get().runtime.output_dir path
    """
    # Access the loss results
    train_losses = evals_result['train']['aft-nloglik']
    val_losses = evals_result['valid']['aft-nloglik']

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
    plt.savefig(os.path.join(output_path, 'losses.png'), dpi=200)
    plt.close()

def survival_curves(time_grid, predicted_medians, sigma=1.0, distribution='normal'):
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

def plot_survival_curves(surv_probs, time_grid, target, output_path) -> None:
    """
    Plot survival curves for all observations.

    Parameters
    ----------
    surv_probs : np.array
        A 2D array of shape (n_obs, n_timepoints) where each row is the survival curve
        for one observation.
    time_grid : np.array
        1D array of time points at which to evaluate the survival function.
    target : pd.DataFrame
        DataFrame containing the time identifier (0th column) and event indicator (1st column).
    output_path : str
        Path to save the plot.
    """

    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter(label='Observed')
    kmf.fit(durations=target.iloc[:, 0], event_observed=target.iloc[:, 1])
    kmf.survival_function_.plot()
    plt.plot(time_grid, np.mean(surv_probs, axis=0), label='Forecasted', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Probability of Survival')
    plt.title('Forecasted & Observed Survival using Holdout Set')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'survival_curves.png'), dpi=200)
    plt.close()