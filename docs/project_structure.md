# Project Structure

```
project-name/
│
├── data/
│   ├── raw/
│   ├── processed/
│   
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── estimate_effects.py
│   │   └── validate_assumptions.py
│   └── visualization/
│       └── visualize.py
│
├── notebooks/
│   ├── 1.data-exploration.ipynb
│   ├── 2.model-development.ipynb
│   └── 3.results-analysis.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── docs/
│   ├── data_dictionary.md
│   ├── analysis_plan.md
│   └── final_report.md
|   └── CONTRIBUTING.md
│
├── tests/
│
├── environment.yml
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Directory Structure Explanation

- `data/`: Store all data files
  - `raw/`: Original, immutable data
  - `processed/`: Cleaned and processed data

- `src/`: Source code for use in this project
  - `data/`: Scripts to download or generate data
  - `models/`: Scripts to train models and make predictions
  - `visualization/`: Scripts to create exploratory and results visualizations

- `notebooks/`: Jupyter notebooks for exploration and communication

- `results/`: Generated analysis as HTML, PDF, LaTeX, etc.
  - `figures/`: Generated graphics and figures to be used in reporting
  - `tables/`: Generated data tables to be used in reporting

- `docs/`: Documentation for your project
  - `data_dictionary.md`: Definitions of variables, units, etc.
  - `analysis_plan.md`: Pre-specified analysis plan
  - `final_report.md`: Final analysis report

- `tests/`: Unit tests for your project

- `environment.yml`: The conda environment file for reproducing the analysis environment
- `requirements.txt`: The requirements file for reproducing the analysis environment
- `README.md`: The top-level description of your project
- `.gitignore`: Specifies intentionally untracked files to ignore

