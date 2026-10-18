# Dataset Placement

Do not generate substitute data. Obtain the processed AMLSim dataset and its manifest/split configuration from the project team or its approved storage. Place these six files together in `data/processed/amlsim_1m/`:

- `transactions.parquet`
- `labels.parquet`
- `splits.parquet`
- `accounts.parquet`
- `manifest.json`
- `split_config.json`

The notebook can also upload a ZIP containing these files, or read them from a mounted Google Drive folder. The notebook validates the file set, parquet readability, minimum columns, unique/non-null transaction IDs, join coverage, label values, feature values, split labels, chronological boundaries, and split configuration before training.

Dataset files are intentionally excluded from Git. Do not commit the AMLSim dataset unless the project owners explicitly authorize it.
