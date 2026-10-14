# Customers Like You — Mini-Project 01

Data engineering and machine learning code for the Online Retail II dataset. Implements **DE1–DE6, ML1–ML7, J1** from the project brief.

## Project success targets (J1)

1. Measure how much smaller the compressed factor representation is than the dense customer × product matrix.
2. Measure fitting, all-products nearest-neighbour generation, and eligible-customer scoring time (excluding raw Excel reading and disk I/O).
3. Compare SVD and PCA hit rate@10 against training-only best sellers using unseen future purchases.
4. Produce a two-dimensional, non-identifying customer map with 4 named and described segments.


## Files

- `build_matrix.py` — DE1–DE6: load both years, profile, sequential cleaning log, time split, holdout, and sparse matrix.
- `analyze.py` — ML1–ML7: SVD k-curve, cosine recommendations, held-out hit@10, PCA comparison, customer segments, sensitivity.
- `analysis_notebook.ipynb` — readable notebook that runs the scripts and reviews outputs.
- `METHOD_DECISION.md` — short method-choice and cleaning justification; review/edit it before submitting.
- `requirements.txt` — installable dependencies.
- `data/raw/` — put the UCI raw workbook or ZIP here; generated data is ignored by Git.
- `data/processed/` — matrices and anonymous indices after processing; internal customer-ID mapping is **not for sharing**.
- `results/` — profile, logging, method comparison and evaluation CSV/JSON.
- `figures/` — k-error plot, hit-rate comparison, and anonymous customer map.

## Setup (Python 3.11+)

```bash
python -m pip install -r requirements.txt
```

Download the dataset from https://archive.ics.uci.edu/dataset/502/online+retail+ii . Place either `online_retail_II.xlsx` or `online+retail+ii.zip` into `data/raw/`. 

## Run

From the folder containing `build_matrix.py` and `analyze.py`:

```bash
python build_matrix.py
python analyze.py
```

Optional notebook:

```bash
jupyter notebook analysis_notebook.ipynb
```



## Definition of holdout and metrics

- **Training:** cleaned invoice lines strictly *before* `latest transaction timestamp − 2 calendar months`.
- **Holdout:** later lines for training-period customers and training-catalogue items, excluding customer–product combinations already purchased during training.
- **Hit rate@10:** fraction of eligible test customers whose personalized top 10 includes at least one held-out product, not recall across all test items.
- **Best-seller baseline:** global ten most purchased products in the training period, omitting products already seen by the customer and backfilling to ten as needed.
- **Matrix values:** binary purchase history (one customer/product purchase flag); alternatives analyzed in sensitivity comparison.
- **Relative reconstruction error:** `||A − A_k||_F / ||A||_F`. PCA reconstruction includes the mean profile.
- **Reported fit memory:** approximately sampled peak process resident memory while fitting (RSS, not just estimator memory). Machine-dependent and sampling may miss short allocation spikes.
- **Reported factor size:** customer and product factors represented as 32-bit floats. This is a *representation size* estimate, not necessarily a compressed archive size.

The demo output includes `k=50`; k is a pragmatic, pre-selected compromise rather than a rank chosen after looking at held-out hit rates. If you later choose k through validation, create another validation split; do not tune on the final test set.

## Team rule

I worked alone for this project

