# PAL: Playtime-aware Attentive Learning for Game Bundle Recommendation

PAL is a Steam bundle recommendation project built on a cross-view bundle recommendation baseline. The repository keeps the training pipeline focused and self-contained so it can be maintained as a standalone project.

This repository is developed as a project for the National University of Singapore (NUS) School of Computing `BT4222` course.

The current implementation includes:

- a `Steam` experiment setting with `topk: [1, 2, 5, 10, 20]`
- user-aware item attention in the item view
- adaptive fusion between item-view and bundle-view scores
- extended ranking metrics: recall, precision, ndcg, hit rate, MAP, MRR, and F1
- checkpoint export and explanation export scripts

## Repository Layout

```text
repo-root/
  docs/                    architecture and experiment notes
  models/                  model definition
  scripts/                 experiment helpers
  train.py                 training and evaluation entry point
  utility.py               dataset loading and metric utilities
  export_explanations.py   explanation export script
  prepare_steam_pal.py     Steam preprocessing helper
  config.yaml              experiment configuration
  datasets/                CSV datasets included with the repository
  presentation/            slide deck sources
  requirements.txt         Python dependencies
```

Generated experiment outputs are written to:

```text
checkpoints/
log/
runs/
results/
wandb/
```

These directories are intentionally ignored for a clean public repository.

## Setup

Create an environment and install dependencies:

```bash
cd PAL-public
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Training expects prepared files under `./datasets/<dataset_name>/`.

For `Steam`, the required files are:

```text
Steam_data_size.csv
bundle_item.csv
user_bundle_train.csv
user_bundle_tune.csv
user_bundle_test.csv
user_item.csv
user_id_map.csv
bundle_id_map.csv
item_id_map.csv
```

See [datasets/README.md](./datasets/README.md) for the expected layout.

## Training

Run from the repository root:

```bash
python train.py -g 0 -m PAL -d Steam -i step0_baseline
```

Useful variants:

```bash
python train.py -g cpu -m PAL -d Steam -i cpu_debug
python train.py -g 0 -m PAL -d Steam -i no_attention --attention-type none --fusion-type none
python train.py -g 0 -m PAL -d Steam -i user_attention --attention-type user --fusion-type none
python train.py -g 0 -m PAL -d Steam -i full_model --attention-type user --fusion-type user
```

`scripts/run_pal_experiments.sh` wraps the common experiment presets.

## Outputs

Training creates:

```text
log/<dataset>/PAL/                     text logs
runs/<dataset>/PAL/                    TensorBoard events
checkpoints/<dataset>/PAL/model/       saved weights
checkpoints/<dataset>/PAL/conf/        saved config JSON
results/<dataset>_PAL_<experiment>.csv metric summaries
```

## Explanations

After training, recommendation explanations can be exported from a saved checkpoint:

```bash
python export_explanations.py \
  --checkpoint checkpoints/Steam/PAL/model/<checkpoint_name> \
  --conf checkpoints/Steam/PAL/conf/<checkpoint_name> \
  --topn 5 \
  --top-items 3 \
  --split test
```

## Notes

- `docs/ARCHITECTURE.md` describes the current PAL model design.
- `docs/IMPROVEMENT_PLAN.md` records the experiment roadmap used during project development.
- `presentation/` contains slide sources only; build artifacts are ignored.
- `docs/PUBLIC_REPO_GUIDE.md` lists what is safe to publish and what should stay private.
- The bundled `Steam` dataset follows the BRUCE index space, so its ID maps are index-to-index CSVs for explanation export compatibility.
