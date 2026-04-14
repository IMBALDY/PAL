#!/usr/bin/env python3
"""Build PAL-format Steam datasets with playtime-aware bundle affinity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Directory containing bundle_data.jsonl and australian_users_items.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/SteamAffinity"))
    parser.add_argument("--dataset-name", default="SteamAffinity")
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--max-bundles", type=int, default=120)
    parser.add_argument("--min-bundle-items", type=int, default=2)
    parser.add_argument("--min-overlap", type=int, default=2)
    parser.add_argument(
        "--min-affinity",
        type=float,
        default=0.0,
        help="Optional hard affinity filter. Keep 0.0 for feature augmentation without extra filtering.",
    )
    parser.add_argument("--min-user-bundles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_parsed_dir(parsed_dir: Path | None) -> Path:
    if parsed_dir is not None:
        return parsed_dir

    candidates = [
        Path("datasets/parsed"),
        Path("../datasets/parsed"),
        Path("steam-review-and-bundle-dataset/parsed"),
        Path("../steam-review-and-bundle-dataset/parsed"),
    ]
    for candidate in candidates:
        if (candidate / "bundle_data.jsonl").exists() and (candidate / "australian_users_items.jsonl").exists():
            return candidate

    raise FileNotFoundError("Could not find parsed Steam data.")


def load_bundles(path: Path, max_bundles: int, min_bundle_items: int) -> tuple[list[dict], set[str]]:
    bundles: list[dict] = []
    item_universe: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            item_ids = []
            item_names = {}
            item_genres = {}
            seen = set()

            for item in raw.get("items", []):
                item_id = item.get("item_id")
                if item_id and item_id not in seen:
                    seen.add(item_id)
                    item_ids.append(item_id)
                    item_names[item_id] = item.get("item_name", "")
                    item_genres[item_id] = item.get("genre", "")

            if len(item_ids) < min_bundle_items:
                continue

            bundles.append(
                {
                    "bundle_id": raw["bundle_id"],
                    "bundle_name": raw.get("bundle_name", ""),
                    "item_ids": item_ids,
                    "item_names": item_names,
                    "item_genres": item_genres,
                }
            )
            item_universe.update(item_ids)
            if len(bundles) >= max_bundles:
                break

    if not bundles:
        raise ValueError("No usable bundles found")
    return bundles, item_universe


def normalized_log_playtimes(items: list[dict], item_universe: set[str]) -> tuple[dict[str, float], dict[str, int]]:
    raw_playtimes = {
        item["item_id"]: int(item.get("playtime_forever", 0))
        for item in items
        if item.get("item_id") in item_universe and int(item.get("playtime_forever", 0)) > 0
    }
    if not raw_playtimes:
        return {}, {}

    log_playtimes = {item_id: math.log1p(playtime) for item_id, playtime in raw_playtimes.items()}
    max_log_playtime = max(log_playtimes.values())
    normalized = {item_id: value / max_log_playtime for item_id, value in log_playtimes.items()}
    return normalized, raw_playtimes


def score_bundle_affinity(user_item_weights: dict[str, float], bundle_items: set[str]) -> tuple[int, float]:
    overlap_items = user_item_weights.keys() & bundle_items
    overlap_count = len(overlap_items)
    affinity = sum(user_item_weights[item_id] for item_id in overlap_items) / len(bundle_items)
    return overlap_count, affinity


def load_users(
    path: Path,
    bundles: list[dict],
    item_universe: set[str],
    max_users: int,
    min_overlap: int,
    min_affinity: float,
    min_user_bundles: int,
) -> tuple[list[dict], set[str]]:
    selected_users: list[dict] = []
    used_items: set[str] = set()
    bundle_item_sets = [set(bundle["item_ids"]) for bundle in bundles]

    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            user_item_weights, raw_playtimes = normalized_log_playtimes(raw.get("items", []), item_universe)
            if not user_item_weights:
                continue

            positive_bundles = []
            for bundle_idx, bundle_items in enumerate(bundle_item_sets):
                overlap_count, affinity = score_bundle_affinity(user_item_weights, bundle_items)
                if overlap_count >= min_overlap and affinity >= min_affinity:
                    positive_bundles.append(
                        {
                            "bundle_idx": bundle_idx,
                            "overlap_count": overlap_count,
                            "affinity": affinity,
                        }
                    )

            if len(positive_bundles) < min_user_bundles or len(positive_bundles) >= len(bundles):
                continue

            selected_users.append(
                {
                    "user_id": raw["user_id"],
                    "owned_items": sorted(user_item_weights),
                    "user_item_weights": user_item_weights,
                    "raw_playtimes": raw_playtimes,
                    "positive_bundles": positive_bundles,
                }
            )
            used_items.update(user_item_weights)
            if len(selected_users) >= max_users:
                break

    if not selected_users:
        raise ValueError("No usable users found; relax affinity/user filters")
    return selected_users, used_items


def write_pairs(path: Path, pairs: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_idx", "right_idx"])
        for left, right in sorted(pairs):
            writer.writerow([left, right])


def write_affinity_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_idx", "bundle_idx", "overlap_count", "affinity"])
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_user_item_weights(path: Path, rows: list[tuple[int, int, float, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_idx", "item_idx", "weight", "raw_playtime"])
        for user_idx, item_idx, weight, raw_playtime in sorted(rows):
            writer.writerow([user_idx, item_idx, f"{weight:.8f}", raw_playtime])


def write_data_size(path: Path, num_users: int, num_bundles: int, num_items: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_users", "num_bundles", "num_items"])
        writer.writerow([num_users, num_bundles, num_items])


def write_id_map(path: Path, id_map: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_id", "inner_id"])
        for original_id, inner_id in sorted(id_map.items(), key=lambda item: item[1]):
            writer.writerow([original_id, inner_id])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    parsed_dir = resolve_parsed_dir(args.parsed_dir)

    bundles, item_universe = load_bundles(
        parsed_dir / "bundle_data.jsonl",
        args.max_bundles,
        args.min_bundle_items,
    )
    users, used_items = load_users(
        parsed_dir / "australian_users_items.jsonl",
        bundles,
        item_universe,
        args.max_users,
        args.min_overlap,
        args.min_affinity,
        args.min_user_bundles,
    )

    item_ids = sorted(set().union(*(set(bundle["item_ids"]) for bundle in bundles)) | used_items)
    item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    bundle_map = {bundle["bundle_id"]: idx for idx, bundle in enumerate(bundles)}
    user_map = {user["user_id"]: idx for idx, user in enumerate(users)}

    bundle_item_pairs = [
        (bundle_idx, item_map[item_id])
        for bundle_idx, bundle in enumerate(bundles)
        for item_id in bundle["item_ids"]
    ]
    user_item_pairs = [
        (user_idx, item_map[item_id])
        for user_idx, user in enumerate(users)
        for item_id in user["owned_items"]
    ]
    user_item_weight_rows = [
        (
            user_idx,
            item_map[item_id],
            user["user_item_weights"][item_id],
            user["raw_playtimes"][item_id],
        )
        for user_idx, user in enumerate(users)
        for item_id in user["owned_items"]
    ]

    user_bundle_train: list[tuple[int, int]] = []
    user_bundle_tune: list[tuple[int, int]] = []
    user_bundle_test: list[tuple[int, int]] = []
    affinity_records: list[dict] = []

    for user_idx, user in enumerate(users):
        positives = list(user["positive_bundles"])
        rng.shuffle(positives)
        user_bundle_tune.append((user_idx, positives[0]["bundle_idx"]))
        user_bundle_test.append((user_idx, positives[1]["bundle_idx"]))
        user_bundle_train.extend((user_idx, positive["bundle_idx"]) for positive in positives[2:])

        for positive in positives:
            affinity_records.append(
                {
                    "user_idx": user_idx,
                    "bundle_idx": positive["bundle_idx"],
                    "overlap_count": positive["overlap_count"],
                    "affinity": positive["affinity"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(args.output_dir / "bundle_item.csv", bundle_item_pairs)
    write_pairs(args.output_dir / "user_item.csv", user_item_pairs)
    write_user_item_weights(args.output_dir / "user_item_weight.csv", user_item_weight_rows)
    write_pairs(args.output_dir / "user_bundle_train.csv", user_bundle_train)
    write_pairs(args.output_dir / "user_bundle_tune.csv", user_bundle_tune)
    write_pairs(args.output_dir / "user_bundle_test.csv", user_bundle_test)
    write_affinity_records(args.output_dir / "user_bundle_affinity.csv", affinity_records)

    write_data_size(args.output_dir / f"{args.dataset_name}_data_size.csv", len(users), len(bundles), len(item_ids))
    write_id_map(args.output_dir / "user_id_map.csv", user_map)
    write_id_map(args.output_dir / "bundle_id_map.csv", bundle_map)
    write_id_map(args.output_dir / "item_id_map.csv", item_map)

    print(
        "Generated {name}: users={users}, bundles={bundles}, items={items}, "
        "train={train}, tune={tune}, test={test}, user_item={user_item}, user_item_weight={user_item_weight}, "
        "bundle_item={bundle_item}, affinity_edges={affinity_edges}, "
        "min_overlap={min_overlap}, min_affinity={min_affinity}".format(
            name=args.dataset_name,
            users=len(users),
            bundles=len(bundles),
            items=len(item_ids),
            train=len(user_bundle_train),
            tune=len(user_bundle_tune),
            test=len(user_bundle_test),
            user_item=len(user_item_pairs),
            user_item_weight=len(user_item_weight_rows),
            bundle_item=len(bundle_item_pairs),
            affinity_edges=len(affinity_records),
            min_overlap=args.min_overlap,
            min_affinity=args.min_affinity,
        )
    )


if __name__ == "__main__":
    main()
