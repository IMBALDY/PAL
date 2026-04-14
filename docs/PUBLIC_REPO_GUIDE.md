# Public Repo Guide

This file defines what should be included in a public `PAL` repository and what must stay private.

## Safe To Upload

- `README.md`
- `requirements.txt`
- `config.yaml`
- `train.py`
- `utility.py`
- `export_explanations.py`
- `prepare_steam_pal.py`
- `models/`
- `scripts/`
- `docs/`
- `datasets/`
- `presentation/main.tex`
- `presentation/README.md`
- `.gitignore`

## Do Not Upload

- `dataset.tgz`
  Local archive only.
- `checkpoints/`
  Trained model artifacts are experiment outputs.
- `runs/`
  TensorBoard logs are generated outputs.
- `log/`
  Training logs are generated outputs.
- `results/`
  Exported metrics can contain local experiment history you may not want to publish.
- `wandb/`
  Run metadata can expose local machine details and account-level information.
- `presentation/*.pdf`
  Exported slide PDFs are generated artifacts.
- `presentation/speaker_notes_*`
  Private speaking notes should stay local unless you explicitly want to publish them.
- `__pycache__/`, `.DS_Store`, `.venv/`
  Local environment and cache files.

## Privacy Checks Before Publishing

- Search for absolute local paths such as `/Users/...` or `/home/...`.
- Search for personal names, usernames, emails, or machine names.
- Search for tokens, API keys, secrets, and credentials.
- Review `wandb/`, logs, and generated configs before copying anything manually.

## Recommended Public Repo Shape

```text
repo-root/
  docs/
  datasets/
  models/
  presentation/
    README.md
    main.tex
  scripts/
  .gitignore
  README.md
  config.yaml
  export_explanations.py
  prepare_steam_pal.py
  requirements.txt
  train.py
  utility.py
```

## Copy Strategy

When creating the public repo, copy only the files and folders listed in `Safe To Upload`.

Do not initialize the public repository by copying the whole current working directory first and deleting afterward. That is the easiest way to leak private files by mistake.
