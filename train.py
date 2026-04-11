#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
from datetime import datetime
from itertools import product

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from utility import Datasets
from models.PAL import PAL


MODEL_NAME = "PAL"
METRIC_NAMES = ["recall", "precision", "ndcg", "hit_rate", "map", "mrr", "f1"]
RESULT_COLUMNS = ["timestamp", "experiment_name", "dataset", "model", "epoch", "step", "split", "topk", "is_best"] + METRIC_NAMES


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="SteamDebug", type=str, help="which dataset to use, options: SteamDebug, NetEase, Youshu, iFashion")
    parser.add_argument("-m", "--model", default=MODEL_NAME, type=str, help="which model to use, options: PAL")
    parser.add_argument("-i", "--info", default="", type=str, help="auxiliary info appended to the log file name")
    parser.add_argument("--attention-type", default=None, choices=["global", "user", "none"], help="override item attention mode")
    parser.add_argument("--fusion-type", default=None, choices=["global", "user", "none"], help="override cross-view fusion mode")
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="BT4222-PAL", type=str, help="wandb project name")
    parser.add_argument("--wandb-entity", default=None, type=str, help="optional wandb entity/team")
    parser.set_defaults(use_wandb=True)
    args = parser.parse_args()

    return args


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
    with open("./config.yaml") as f:
        conf = yaml.safe_load(f)
    print("load config file done!", flush=True)

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    print(f"[train.py] args={paras}", flush=True)

    assert paras["model"] in [MODEL_NAME], "Pls select models from: PAL"

    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    conf["use_wandb"] = paras["use_wandb"]
    conf["wandb_project"] = paras["wandb_project"]
    conf["wandb_entity"] = paras["wandb_entity"]

    if paras["attention_type"] is not None:
        conf["use_item_attention"] = paras["attention_type"] != "none"
        if conf["use_item_attention"]:
            conf["attention_type"] = paras["attention_type"]
    if paras["fusion_type"] is not None:
        conf["use_view_fusion"] = paras["fusion_type"] != "none"
        if conf["use_view_fusion"]:
            conf["fusion_type"] = paras["fusion_type"]

    dataset = Datasets(conf)
    print(
        f"[train.py] dataset_loaded dataset={dataset_name} users={dataset.num_users} bundles={dataset.num_bundles} items={dataset.num_items} "
        f"train_batches={len(dataset.train_loader)} val_batches={len(dataset.val_loader)} test_batches={len(dataset.test_loader)}",
        flush=True,
    )

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    device = resolve_device(conf["gpu"])
    conf["device"] = device
    print(f"[train.py] device={device}", flush=True)
    print(conf, flush=True)

    for lr, l2_reg, item_level_ratio, bundle_level_ratio, bundle_agg_ratio, embedding_size, num_layers, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['item_level_ratios'], conf['bundle_level_ratios'], conf['bundle_agg_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]):
        log_path = f"./log/{conf['dataset']}/{conf['model']}"
        run_path = f"./runs/{conf['dataset']}/{conf['model']}"
        checkpoint_model_path = f"./checkpoints/{conf['dataset']}/{conf['model']}/model"
        checkpoint_conf_path = f"./checkpoints/{conf['dataset']}/{conf['model']}/conf"
        result_base_path = f"./results/{conf['dataset']}_{conf['model']}"
        for path in [run_path, log_path, checkpoint_model_path, checkpoint_conf_path, "./results"]:
            os.makedirs(path, exist_ok=True)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert item_level_ratio == 0 and bundle_level_ratio == 0 and bundle_agg_ratio == 0

        settings += [f"Neg_{conf['neg_num']}", str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)]

        conf["item_level_ratio"] = item_level_ratio
        conf["bundle_level_ratio"] = bundle_level_ratio
        conf["bundle_agg_ratio"] = bundle_agg_ratio
        conf["num_layers"] = num_layers
        settings += [str(item_level_ratio), str(bundle_level_ratio), str(bundle_agg_ratio), str(num_layers)]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]

        attention_tag = "attn_none"
        if conf.get("use_item_attention", True):
            attention_tag = f"attn_{conf.get('attention_type', 'user')}"
        fusion_tag = "fusion_none"
        if conf.get("use_view_fusion", True):
            fusion_tag = f"fusion_{conf.get('fusion_type', 'user')}"
        settings += [attention_tag, fusion_tag]

        setting = "_".join(settings)
        print(f"[train.py] START setting={setting}", flush=True)
        result_path = f"{result_base_path}_{setting}_results.csv"
        log_path = f"{log_path}/{setting}"
        run_path = f"{run_path}/{setting}"
        checkpoint_model_path = f"{checkpoint_model_path}/{setting}"
        checkpoint_conf_path = f"{checkpoint_conf_path}/{setting}"
            
        run = SummaryWriter(run_path)
        wandb_run = init_wandb(conf, setting)

        if conf['model'] == MODEL_NAME:
            model = PAL(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))
        print(f"[train.py] model_initialized setting={setting}", flush=True)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])
        print(
            f"[train.py] training_schedule setting={setting} epochs={conf['epochs']} batch_cnt={batch_cnt} "
            f"test_interval_bs={test_interval_bs} ed_interval_bs={ed_interval_bs}",
            flush=True,
        )

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            print(f"[train.py] epoch_start setting={setting} epoch={epoch}", flush=True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor+1) % ed_interval_bs == 0:
                    ED_drop = True
                bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                loss = bpr_loss + conf["c_lambda"] * c_loss
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)
                log_wandb_train(
                    wandb_run,
                    epoch,
                    batch_anchor,
                    loss_scalar.item(),
                    bpr_loss_scalar.item(),
                    c_loss_scalar.item(),
                )

                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                if (batch_anchor+1) % test_interval_bs == 0:  
                    print(f"[train.py] eval_start setting={setting} epoch={epoch} step={batch_anchor}", flush=True)
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, wandb_run, log_path, result_path, setting, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)
                    print(f"[train.py] eval_end setting={setting} epoch={epoch} step={batch_anchor}", flush=True)

        if wandb_run is not None:
            wandb_run.finish()
            print(f"[train.py] wandb_finished setting={setting}", flush=True)
        print(f"[train.py] END setting={setting}", flush=True)


def init_wandb(conf, setting):
    if not conf.get("use_wandb", True):
        print(f"[train.py] wandb_disabled setting={setting}", flush=True)
        return None
    if wandb is None:
        raise ImportError("wandb is enabled but not installed in the current environment")

    config_for_wandb = {}
    for key, value in conf.items():
        if key == "device":
            continue
        if isinstance(value, (str, int, float, bool, list, dict)):
            config_for_wandb[key] = value
        else:
            config_for_wandb[key] = str(value)

    run = wandb.init(
        project=conf.get("wandb_project", "BT4222-PAL"),
        entity=conf.get("wandb_entity"),
        name=setting,
        group=f"{conf['dataset']}_{conf['model']}",
        job_type="train",
        config=config_for_wandb,
        reinit=True,
    )
    print(
        f"[train.py] wandb_initialized setting={setting} project={conf.get('wandb_project', 'BT4222-PAL')} "
        f"entity={conf.get('wandb_entity')}",
        flush=True,
    )
    return run


def log_wandb_train(wandb_run, epoch, step, loss, bpr_loss, c_loss):
    if wandb_run is None:
        return
    wandb_run.log(
        {
            "train/epoch": epoch,
            "train/step": step,
            "train/loss": loss,
            "train/loss_bpr": bpr_loss,
            "train/loss_c": c_loss,
        },
        step=step,
    )


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        for metric in METRIC_NAMES:
            best_metrics[key][metric] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_metric_str = ", ".join([f"{m}: {val_scores[m][topk]:f}" for m in METRIC_NAMES])
    test_metric_str = ", ".join([f"{m}: {test_scores[m][topk]:f}" for m in METRIC_NAMES])
    val_str = f"{curr_time}, Top_{topk}, Val:  {val_metric_str}"
    test_str = f"{curr_time}, Top_{topk}, Test: {test_metric_str}"

    with open(log_path, "a") as log:
        log.write(f"{val_str}\n")
        log.write(f"{test_str}\n")

    print(val_str)
    print(test_str)


def append_results_csv(conf, result_path, experiment_name, metrics, epoch, step, is_best):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(result_path) or os.path.getsize(result_path) == 0

    with open(result_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for split, split_metrics in metrics.items():
            for topk in conf["topk"]:
                row = {
                    "timestamp": timestamp,
                    "experiment_name": experiment_name,
                    "dataset": conf["dataset"],
                    "model": conf["model"],
                    "epoch": epoch,
                    "step": step,
                    "split": split,
                    "topk": topk,
                    "is_best": is_best,
                }
                for metric in METRIC_NAMES:
                    row[metric] = split_metrics[metric][topk]
                writer.writerow(row)


def log_metrics(conf, model, metrics, run, wandb_run, log_path, result_path, experiment_name, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    with torch.no_grad():
        gate_stats = model.get_gate_statistics(model.propagate(test=True))
    run.add_scalar("fusion/mean_beta", gate_stats["mean_beta"], batch_anchor)
    run.add_scalar("fusion/std_beta", gate_stats["std_beta"], batch_anchor)
    with open(log_path, "a") as log:
        log.write(
            f"FusionGate step={batch_anchor}: mean_beta={gate_stats['mean_beta']:.6f}, "
            f"std_beta={gate_stats['std_beta']:.6f}, min_beta={gate_stats['min_beta']:.6f}, "
            f"max_beta={gate_stats['max_beta']:.6f}\n"
        )

    topk_ = 20
    print(f"top{topk_} as the final evaluation standard")
    is_best = metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]
    log_wandb_eval(wandb_run, metrics, gate_stats, epoch, batch_anchor, is_best)
    append_results_csv(conf, result_path, experiment_name, metrics, epoch, batch_anchor, is_best)
    if is_best:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        with open(checkpoint_conf_path, "w") as f:
            json.dump(dump_conf, f)
        print(
            f"[train.py] new_best setting={experiment_name} epoch={epoch} step={batch_anchor} "
            f"checkpoint_model={checkpoint_model_path} checkpoint_conf={checkpoint_conf_path}",
            flush=True,
        )
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a") as log:
            for topk in conf['topk']:
                for key, res in best_metrics.items():
                    for metric in res:
                        best_metrics[key][metric][topk] = metrics[key][metric][topk]

                test_metric_str = ", ".join([f"{m.upper()}_T={best_metrics['test'][m][topk]:.5f}" for m in METRIC_NAMES])
                val_metric_str = ", ".join([f"{m.upper()}_V={best_metrics['val'][m][topk]:.5f}" for m in METRIC_NAMES])
                best_perform["test"][topk] = f"{curr_time}, Best in epoch {best_epoch}, TOP {topk}: {test_metric_str}"
                best_perform["val"][topk] = f"{curr_time}, Best in epoch {best_epoch}, TOP {topk}: {val_metric_str}"
                print(best_perform["val"][topk])
                print(best_perform["test"][topk])
                log.write(best_perform["val"][topk] + "\n")
                log.write(best_perform["test"][topk] + "\n")

    return best_metrics, best_perform, best_epoch


def log_wandb_eval(wandb_run, metrics, gate_stats, epoch, step, is_best):
    if wandb_run is None:
        return

    payload = {
        "eval/epoch": epoch,
        "eval/step": step,
        "eval/is_best": int(is_best),
        "fusion/mean_beta": gate_stats["mean_beta"],
        "fusion/std_beta": gate_stats["std_beta"],
        "fusion/min_beta": gate_stats["min_beta"],
        "fusion/max_beta": gate_stats["max_beta"],
    }
    for split, split_metrics in metrics.items():
        for metric_name, metric_values in split_metrics.items():
            for topk, value in metric_values.items():
                payload[f"{split}/{metric_name}@{topk}"] = value

    wandb_run.log(payload, step=step)


def test(model, dataloader, conf):
    print(f"[train.py] test_loop_start batches={len(dataloader)}", flush=True)
    tmp_metrics = {}
    for m in METRIC_NAMES:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    with torch.no_grad():
        rs = model.propagate(test=True)
        for users, ground_truth_u_b, train_mask_u_b in dataloader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8 * train_mask_u_b.to(device)
            tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    print(f"[train.py] test_loop_end metrics_ready topk={conf['topk']}", flush=True)
    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {m: {} for m in METRIC_NAMES}
    # 确保 grd 和 pred 在同一设备上
    grd = grd.to(pred.device)
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp_topk = get_topk_metrics(grd, is_hit, topk)
        for m in METRIC_NAMES:
            tmp[m][topk] = tmp_topk[m]

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_topk_metrics(grd, is_hit, topk):
    epsilon = 1e-8
    device = grd.device
    num_pos = grd.sum(dim=1)
    valid_user = num_pos > 0
    denorm = valid_user.sum().item()
    if denorm == 0:
        return {m: [0, 0] for m in METRIC_NAMES}

    is_hit = is_hit.float()
    hit_cnt = is_hit.sum(dim=1)
    recall = hit_cnt / (num_pos + epsilon)
    precision = hit_cnt / topk
    hit_rate = (hit_cnt > 0).float()
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    rank_positions = torch.arange(1, topk + 1, device=device, dtype=torch.float)
    discounts = torch.log2(rank_positions + 1)
    dcg = (is_hit / discounts).sum(dim=1)
    num_pos_at_k = num_pos.clamp(0, topk).to(torch.long)
    ideal_hits = torch.arange(topk, device=device).unsqueeze(0) < num_pos_at_k.unsqueeze(1)
    idcg = (ideal_hits.float() / discounts).sum(dim=1)
    ndcg = dcg / (idcg + epsilon)

    precision_at_rank = is_hit.cumsum(dim=1) / rank_positions
    ap = (precision_at_rank * is_hit).sum(dim=1) / torch.minimum(num_pos, torch.tensor(topk, device=device, dtype=num_pos.dtype)).clamp(min=1)
    reciprocal_rank = (is_hit / rank_positions).max(dim=1).values

    return {
        "recall": [recall[valid_user].sum().item(), denorm],
        "precision": [precision[valid_user].sum().item(), denorm],
        "ndcg": [ndcg[valid_user].sum().item(), denorm],
        "hit_rate": [hit_rate[valid_user].sum().item(), denorm],
        "map": [ap[valid_user].sum().item(), denorm],
        "mrr": [reciprocal_rank[valid_user].sum().item(), denorm],
        "f1": [f1[valid_user].sum().item(), denorm],
    }


if __name__ == "__main__":
    main()
