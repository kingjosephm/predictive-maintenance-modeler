# Predictive Maintenance Modeler

Predictive Maintenance Modeler using [XGBoost's Accelerated Failure Time Implementation](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html). This is a survival analysis model (see explanation below), applied to the automotive or industrial context. This is a personal project inspired by a work project; please feel free to fork the repository or contact me if you have suggestions.


# Modeling Motivation
In many real world settings we are interested in modeling the duration of time until an event occurs, whether that event be a mechanical failure in the automotive context, death in a biological sense, or a customer purchase of a good or service (churn). The type of modeling context is known as [survival analysis in statistics, reliability analysis in engineering, or duration analysis in economics](https://en.wikipedia.org/wiki/Survival_analysis).

Survival analysis seeks to model the time to a *singlar* event for an entity (e.g. a human or machine). However, in many real-life situations, entitles can potentially experience *multiple* types of event, e.g. a person can die from cancer, a heart attack, car accident. When only one of these types of events can occur this is referred to a **competing events**, in the sense that each event competes with the other to yield the observed event of interest, and the occurrence of one type of event precludes the occurrence of the others. Consequently, probabilities of each events are referred to as **competing risks** (for more information, [see](https://www.publichealth.columbia.edu/research/population-health-methods/competing-risk-analysis)).

***The code in this repository was developed to model a single outcome for classicial time-to-failure analysis. Competing events is currently not supported.***

## Censoring
Fundamental to both classical survival and competing events models is the idea of **censoring**, i.e. when the event(s) of interest is not observed for some subjects. These subjects are referred to as "right-censored", as their true lifetimes are to the right of the observed timeline, either because the study ended before an event, or because the subject exited the study early (attrition). Critically, it is assumed that censored is uncorrelated with the outcome(s) of interest. That is, censored observations are assumed to have the same probability of experiencing a subsequent event as those observations whose event was observed (the so-called "non-informative censoring" assumption). To borrow an example from [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html):

![censoring](./assets/survival_analysis_example.png)

## Predictive vs. Preventative Maintenance
In automotive and manufacturing applications, we are frequently interested anticipating failures before they occur. There are several approaches to this, sorted by increasing level of modeling sophistication:

1) **Reactive maintenance** - perform maintenance only when a component breaks
2) **Preventative maintenance** - perform maintenance based on a maintenance (time or usage) schedules, regardless of the actual physical wear
3) **Conditions-based maintenance** - perform maintenance based on *current* wear indications, but the actual remaining useful life (RuL) is unclear
4) **Predictive maintenance** - perform maintenance using a forecast to estimate the RuL

Of these strategies, predictive mainteance is the most efficient as it promises to replace parts or components *just in time*, optimizing their lifecycle.

# Quick Start

## Requirements
The Python environment used for this project was created using the Conda package manager, which if you don't already can be downloaded [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). This code has only been developed for local execution. It hasn't been configured for any cloud or Docker implementation, though this would be logical extension.

#### OSX
This code and environment were developed on a Mac. If you also use OSX, you can create the same Python 3.11.8 environment simply by running:

    conda env create -f environment.yaml
    conda activate pdm_v2  # activates the environment

#### Windows

    conda create --name pdm_v2 python=3.12.2  # creates new virtual environment
    conda activate pdm_v2

Windows users should next install the following packages `pdm_v2` virtual environment:
- conda install
  - xgboost
  - hydra-core
  - pandas
  - scikit-learn
  - matplotlib
  - lifelines
  - optuna
  - scikit-survival
- pip install
  - you can ignore these dependencies

This hasn't been tested on Windows, so you may need to experiment a bit.

## Running the Code
Prior to running the code you must configure the training and data build file, which is done in [config.yaml](config.yaml). Currently, this is populated with the default values for the Scania dataset (explained below).

This config file contains the following arguments:
- `data_path`: path to CSV training data. Code expects CSV data type and will throw an error otherwise
- `unit_identifier`: ID feature name
- `time_identifier`: time feature name
- `target_feature`: outcome feature name. Note - this feature is expected to be binary or boolean
- `test_size`: proportion of unique IDs to use in cross-validation. Test set is constructued assuming a panel structure, though works equivalently for cross-sectional data
- `seed`: initializer value for pseudo random number generator
- `lag_length`: only applies to panel data. Number of previous time-series lags to add as new columns *per feature*. Time-series observations < `lag_length` will be filled as far back as possible, using the values for a given ID's first time period to fill the rest
- `sampling_n`: Maximum number of observations per ID to sample. This reduces the influence of statistical outliers (e.g. vehicles with many time-series observations). Set to 1 to convert panel data to a simple cross-section by keeping only the last observation per `unit_identifier`. Note - `sampling_n` must be >= `lag_length`.
- `hyperoptimize`: whether to hyperoptimize using [Optuna](https://optuna.org/). This employs the default [Tree-structured Parzen Estimator](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- `n_trials`: number of hyperparameter trials to execute. Ignored if `hyperoptimize` is *false*
- `save_model`: whether to save trained XGBoost model as JSON at the end


To run the code enter into the terminal:

```
python main.py
```

Logging is streamed to the console during the run procedure. At the end, a log file, figures, and model artefacts are output to a new folder with the datetime when training is initiated: `./output/<YYYY-MM-DD>/<HH-mm-SS>`.

# Example
To see an example of this code applied to a multivariate time-series dataset from Scania trucks, see [scania example](./scania_example/README.md)
