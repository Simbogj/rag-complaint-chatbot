# Data directory

Do **not** commit raw or processed datasets to Git. The CFPB complaints file is ~5.6 GB and will block GitHub pushes.

## Setup

1. Download the CFPB complaint CSV from the challenge resources.
2. Place it at:

```
data/raw/complaints.csv
```

3. Run preprocessing:

```bash
python -m src.preprocess
```

This generates:
- `data/filtered_complaints.csv`
- `data/processed/complaints_clean.csv`

Both paths are gitignored and stay on your machine only.
