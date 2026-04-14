#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from models.PAL import PAL
from utility import Datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Export PAL recommendation explanations from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint saved by train.py")
    parser.add_argument("--conf", required=True, help="Path to checkpoint config JSON saved by train.py")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to PAL/results/explanations_<checkpoint>.csv")
    parser.add_argument("--gpu", default=None, help="Optional GPU override. Use cpu to force CPU inference.")
    parser.add_argument("--topn", type=int, default=5, help="How many recommendations to export per user")
    parser.add_argument("--top-items", type=int, default=3, help="How many explanation items to keep per recommendation")
    parser.add_argument("--history-items", type=int, default=5, help="How many user history items to include")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="Which split to export against")
    return parser.parse_args()


def load_inverse_map(path):
    if not path.exists():
        return {}
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {int(row["inner_id"]): str(row["original_id"]) for row in reader}
    raise ValueError(f"Unsupported map file format: {path}")


def format_ids(ids, inverse_map):
    return "|".join(inverse_map.get(int(idx), str(int(idx))) for idx in ids)


def format_weighted_ids(ids, weights, inverse_map):
    parts = []
    for idx, weight in zip(ids, weights):
        if float(weight) < 0:
            continue
        mapped = inverse_map.get(int(idx), str(int(idx)))
        parts.append(f"{mapped}:{float(weight):.4f}")
    return "|".join(parts)


def get_user_history(dataset, user_idx, history_items):
    user_item_graph = dataset.graphs[1]
    history = user_item_graph[user_idx].indices.tolist()
    return history[:history_items]


def get_ground_truth(dataset, split_name, user_idx):
    if split_name == "test":
        graph = dataset.bundle_test_data.u_b_graph
    else:
        graph = dataset.bundle_val_data.u_b_graph
    return graph[user_idx].indices.tolist()


def resolve_output_path(args):
    if args.output is not None:
        return Path(args.output)
    checkpoint_name = Path(args.checkpoint).name
    return Path("results") / f"explanations_{checkpoint_name}.csv"


def resolve_device(gpu_value):
    if gpu_value == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_value)
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    with open(args.conf, encoding="utf-8") as f:
        conf = json.load(f)

    if args.gpu is not None:
        conf["gpu"] = args.gpu
    gpu_value = conf.get("gpu", "0")
    device = resolve_device(gpu_value)
    conf["device"] = device

    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    model = PAL(conf, dataset.graphs).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset.path) / conf["dataset"]
    user_map = load_inverse_map(dataset_dir / "user_id_map.csv")
    bundle_map = load_inverse_map(dataset_dir / "bundle_id_map.csv")
    item_map = load_inverse_map(dataset_dir / "item_id_map.csv")

    dataloader = dataset.test_loader if args.split == "test" else dataset.val_loader

    fieldnames = [
        "user_idx",
        "user_id",
        "ground_truth_bundle_indices",
        "ground_truth_bundle_ids",
        "history_item_indices",
        "history_item_ids",
        "recommended_bundle_idx",
        "recommended_bundle_id",
        "rank",
        "score",
        "item_view_score",
        "bundle_view_score",
        "fusion_weight",
        "top_item_indices",
        "top_item_ids",
        "top_item_weights",
    ]

    with torch.no_grad():
        propagate_result = model.propagate(test=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for user_indices, _, train_mask in dataloader:
                users = user_indices.to(device)
                scores = model.evaluate(propagate_result, users)
                scores -= 1e8 * train_mask.to(device)
                top_scores, top_bundle_ids = torch.topk(scores, args.topn, dim=1)

                flat_users = users.unsqueeze(1).expand(-1, args.topn).reshape(-1)
                flat_bundles = top_bundle_ids.reshape(-1)
                details = model.explain_recommendation(
                    propagate_result,
                    flat_users,
                    flat_bundles,
                    topk=args.top_items,
                )

                top_item_ids = details["top_item_ids"].cpu().tolist()
                top_item_weights = details["top_item_weights"].cpu().tolist()
                score_values = details["score"].cpu().tolist()
                item_view_scores = details["item_view_score"].cpu().tolist()
                bundle_view_scores = details["bundle_view_score"].cpu().tolist()
                fusion_weights = details["fusion_weight"].cpu().tolist()

                row_count = user_indices.shape[0]
                for row_idx in range(row_count):
                    user_idx = int(user_indices[row_idx].item())
                    user_history = get_user_history(dataset, user_idx, args.history_items)
                    ground_truth = get_ground_truth(dataset, args.split, user_idx)

                    for rank_idx in range(args.topn):
                        flat_idx = row_idx * args.topn + rank_idx
                        bundle_idx = int(top_bundle_ids[row_idx, rank_idx].item())
                        writer.writerow(
                            {
                                "user_idx": user_idx,
                                "user_id": user_map.get(user_idx, str(user_idx)),
                                "ground_truth_bundle_indices": "|".join(str(idx) for idx in ground_truth),
                                "ground_truth_bundle_ids": format_ids(ground_truth, bundle_map),
                                "history_item_indices": "|".join(str(idx) for idx in user_history),
                                "history_item_ids": format_ids(user_history, item_map),
                                "recommended_bundle_idx": bundle_idx,
                                "recommended_bundle_id": bundle_map.get(bundle_idx, str(bundle_idx)),
                                "rank": rank_idx + 1,
                                "score": f"{float(score_values[flat_idx]):.6f}",
                                "item_view_score": f"{float(item_view_scores[flat_idx]):.6f}",
                                "bundle_view_score": f"{float(bundle_view_scores[flat_idx]):.6f}",
                                "fusion_weight": f"{float(fusion_weights[flat_idx]):.6f}",
                                "top_item_indices": "|".join(str(int(idx)) for idx in top_item_ids[flat_idx]),
                                "top_item_ids": format_ids(top_item_ids[flat_idx], item_map),
                                "top_item_weights": format_weighted_ids(top_item_ids[flat_idx], top_item_weights[flat_idx], item_map),
                            }
                        )

    print(f"Saved explanations to {output_path}")


if __name__ == "__main__":
    main()
