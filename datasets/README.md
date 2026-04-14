# Datasets

This directory stores the CSV datasets used by the public PAL repository.

Expected layout:

```text
datasets/
  Steam/
    Steam_data_size.csv
    bundle_item.csv
    user_bundle_train.csv
    user_bundle_tune.csv
    user_bundle_test.csv
    user_item.csv
    user_id_map.csv
    bundle_id_map.csv
    item_id_map.csv
  SteamAffinity/
    ...
```

The training and explanation scripts read from `./datasets/<dataset_name>/`.

Use `prepare_steam_pal.py` to generate PAL-ready dataset files from the raw Steam sources, or copy an already prepared dataset into this folder.
