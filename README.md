DSI23_SPRFitter
==============================

Surface plasmon resonance fitter

Project Organization
----------------------

    ├── Makefile             <- Makefile with commands like `make data` or `make train`
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   ├── external         <- Data from third party sources.
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── processed        <- The final, canonical data sets for modeling.
    │   └── raw              <- The original, immutable data dump.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks            <- Jupyter notebooks. Project notebooks should go to the top-level and
    │   │                       With naming convention a number followed by a delimiter and a description,
    │   │                       e.g. 0X-<description>.py
    │   ├── sandbox          <- Experimental notebooks not part to the project analysis but primarily test of 
    │   │                       explorative settings.
    │   └── templates        <- Template notebooks for copy paste purpose to speed up productivity.
    │       └── 0X_<description>.py
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src/dsi23                <- Source code for use in this project.
    │   └── utils           <- Source code for use in this project
    │       ├── __init__.py <- Makes utils a Python module
    │       ├── data.py     <- Functions to load or generate data
    │       ├── features.py <- Functions to turn raw data into features for modeling
    │       ├── metrics.py  <- Functions to define customized metrics
    │       ├── models.py   <- Functions to train models and perform predictions
    │       ├── viz.py      <- Functions to create exploratory and results oriented visualizations
    │       └── workflow.py <- Functions to ease workflow operations
    │
    ├── tox.ini             <- tox file with settings for running tox; see tox.readthedocs.io
    │
    └── workflow            <- Workflow configuration files such as AirFlow, SnakeMake, Domino etc


--------

<p><small>Project based on the <a target="_blank" href="https://sc216.corp.novocorp.net/modelling/mptprojects/projectcodetemplate.git">cookiecutter NN computational research area project template</a>. #computationalresearch</small></p>
