# Credit Risk Modelling: Empirical Work handover

**Thomas de Lesquen · Felix Sewing · Credit Risk Modelling (MPGI, FGV EAESP; instructor Apinyapon Seingyai, PhD)**

## What's in this folder

| item | what it does |
|---|---|
| `01_DataDownload.ipynb` | downloads and reshapes the two BCB sources, joins them, folds COSIF 1.5 back to 1.0 -> `Output/df_final.parquet`, `df_final_congl.parquet` |
| `02_VariableConstruction.ipynb` | builds the target (BIS breach + explicit default) and the CAMELS ratios -> `df_var.parquet` |
| `03_CategorizationSelection.ipynb` | WoE binning, IV, stepwise logit selection, fit on `train` only -> `df_woe.parquet` |
| `04_ModelDevelopment.ipynb` | logit vs. tree vs. forest, cluster bootstrap, SHAP, the macro overlay and its placebo test -> `df_scores.parquet`, `model_comparison.xlsx` |
| `05_EvaluationMonitoring.ipynb` | discrimination, calibration, the rating scale, grade migration, PSI monitoring -> `ratings.xlsx`, `psi.xlsx` |
| `06_Figures.ipynb` | every chart in the report: the thirteen restyled from notebooks 04/05, plus four built around the macro overlay -> `figures/*.png` |
| `CHANGES_AND_WHY.md` | every place these notebooks  differ from the course notebooks, and why |
| `CRM_Project_Felix_Thomas.pdf` | the submitted report |
| `Input/` | the raw BCB downloads notebook 01 reads, plus the small course reference spreadsheets |
| `Output/` | the full parquet/xlsx chain every notebook reads and writes |
| `figures/` | every chart from notebooks 04-06, `dpi=200` |
| `requirements.txt` | exact package versions this was built and run on |



## Running it

```
pip install -r requirements.txt
```

- Python 3.14. Working directory has to be this folder, every notebook resolves `Input/` and
  `Output/` from `pathlib.Path.cwd()`.

