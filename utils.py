import random
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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