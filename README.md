# GDP Growth Forecasting in Latin America

This repository contains a project focused on comparing different forecasting approaches for GDP growth in Latin American countries.

The study compares:

* ARIMA models
* Machine Learning approaches (Random Forest and XGBoost)
* Functional Data Analysis approaches (FAR(1))

## Repository Contents

* `latin_america_gdp_growth.csv`
  Dataset containing GDP growth data for Latin American countries.

* `models.ipynb`
  Jupyter notebook with the implementation, experiments, and results.

* `models.html`
  HTML export of the notebook for easy visualization without running Jupyter.

* `requirements.txt`
  Python dependencies required to run the project.

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

Open the notebook with Jupyter:

```bash
jupyter notebook models.ipynb
```

Or open `models.html` directly in a web browser to view the results.

## Project Goal

The objective of this project is to evaluate and compare classical statistical methods, machine learning models, and functional approaches for forecasting GDP growth in Latin America.
