#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = -torch.log(torch.sigmoid(pos - negs))
    return torch.mean(loss)


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    return rowsum_sqrt @ graph @ colsum_sqrt


def to_tensor(graph):
    graph = graph.tocoo()
    indices = np.vstack((graph.row, graph.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(graph.data),
        torch.Size(graph.shape),
    ).coalesce()


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    return mask * values


class PAL(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        self.device = self.conf["device"]

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.use_item_attention = conf.get("use_item_attention", True)
        self.attention_type = conf.get("attention_type", "user")
        self.attention_score_type = conf.get("attention_score_type", "dot")
        self.attention_hidden_size = conf.get("attention_hidden_size", self.embedding_size)
        self.use_view_fusion = conf.get("use_view_fusion", True)
        self.fusion_type = conf.get("fusion_type", "user")
        self.fusion_hidden_size = conf.get("fusion_hidden_size", self.embedding_size)
        self.eval_bundle_chunk_size = int(conf.get("eval_bundle_chunk_size", 256))
        self.explain_top_items = int(conf.get("explain_top_items", 3))

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.init_bundle_item_cache()

        # graphs without dropout for evaluation
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # graphs with configured dropouts for training
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()
        self.init_attention_modules()
        self.init_fusion_modules()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)
        self.attention_dropout = nn.Dropout(self.conf.get("attention_dropout", 0.0))

    def init_attention_modules(self):
        if self.attention_type not in {"global", "user"}:
            raise ValueError("attention_type must be one of: global, user")
        if self.attention_score_type not in {"dot", "mlp"}:
            raise ValueError("attention_score_type must be one of: dot, mlp")

        if self.use_item_attention and self.attention_type == "global":
            self.global_attention_proj = nn.Linear(self.embedding_size, 1, bias=False)
        else:
            self.global_attention_proj = None

        if self.use_item_attention and self.attention_type == "user" and self.attention_score_type == "mlp":
            self.user_attention_mlp = nn.Sequential(
                nn.Linear(self.embedding_size * 3, self.attention_hidden_size),
                nn.ReLU(),
                nn.Linear(self.attention_hidden_size, 1),
            )
        else:
            self.user_attention_mlp = None

    def init_fusion_modules(self):
        if self.fusion_type not in {"global", "user"}:
            raise ValueError("fusion_type must be one of: global, user")

        if not self.use_view_fusion:
            self.global_fusion_logit = None
            self.user_fusion_gate = None
            return

        if self.fusion_type == "global":
            self.global_fusion_logit = nn.Parameter(torch.tensor(0.0))
            self.user_fusion_gate = None
        else:
            self.global_fusion_logit = None
            self.user_fusion_gate = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.fusion_hidden_size),
                nn.ReLU(),
                nn.Linear(self.fusion_hidden_size, 1),
            )

    def init_bundle_item_cache(self):
        bi_graph = self.bi_graph.tocsr()
        bundle_sizes = np.diff(bi_graph.indptr).astype(np.int64)
        max_bundle_size = int(bundle_sizes.max()) if len(bundle_sizes) else 1

        bundle_item_ids = np.zeros((self.num_bundles, max_bundle_size), dtype=np.int64)
        bundle_item_mask = np.zeros((self.num_bundles, max_bundle_size), dtype=bool)
        for bundle_idx in range(self.num_bundles):
            start = bi_graph.indptr[bundle_idx]
            end = bi_graph.indptr[bundle_idx + 1]
            item_ids = bi_graph.indices[start:end]
            bundle_item_ids[bundle_idx, : len(item_ids)] = item_ids
            bundle_item_mask[bundle_idx, : len(item_ids)] = True

        self.register_buffer("bundle_item_ids", torch.from_numpy(bundle_item_ids))
        self.register_buffer("bundle_item_mask", torch.from_numpy(bundle_item_mask))
        self.register_buffer("bundle_item_count", torch.from_numpy(bundle_sizes))

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        modification_ratio = self.conf["item_level_ratio"]
        item_level_graph = sp.bmat(
            [
                [sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))],
            ]
        )
        if modification_ratio != 0 and self.conf["aug_type"] == "ED":
            graph = item_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(self.device)

    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        item_level_graph = sp.bmat(
            [
                [sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))],
            ]
        )
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(self.device)

    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        modification_ratio = self.conf["bundle_level_ratio"]
        bundle_level_graph = sp.bmat(
            [
                [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
                [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))],
            ]
        )

        if modification_ratio != 0 and self.conf["aug_type"] == "ED":
            graph = bundle_level_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(self.device)

    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        bundle_level_graph = sp.bmat(
            [
                [sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
                [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))],
            ]
        )
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(self.device)

    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(self.device)

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(self.device)

    def one_propagate(self, graph, a_feature, b_feature, mess_dropout, test):
        features = torch.cat((a_feature, b_feature), 0)
        all_features = [features]

        for layer_idx in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                features = mess_dropout(features)

            features = features / (layer_idx + 2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1).sum(dim=1).squeeze(1)
        return torch.split(all_features, (a_feature.shape[0], b_feature.shape[0]), 0)

    def get_base_bundle_rep(self, item_features, test):
        bundle_agg_graph = self.bundle_agg_graph_ori if test else self.bundle_agg_graph
        bundle_features = torch.matmul(bundle_agg_graph, item_features)
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            bundle_features = self.bundle_agg_dropout(bundle_features)
        return bundle_features

    def propagate(self, test=False):
        if test:
            item_users, item_items = self.one_propagate(
                self.item_level_graph_ori,
                self.users_feature,
                self.items_feature,
                self.item_level_dropout,
                test,
            )
            bundle_users, bundle_bundles = self.one_propagate(
                self.bundle_level_graph_ori,
                self.users_feature,
                self.bundles_feature,
                self.bundle_level_dropout,
                test,
            )
        else:
            item_users, item_items = self.one_propagate(
                self.item_level_graph,
                self.users_feature,
                self.items_feature,
                self.item_level_dropout,
                test,
            )
            bundle_users, bundle_bundles = self.one_propagate(
                self.bundle_level_graph,
                self.users_feature,
                self.bundles_feature,
                self.bundle_level_dropout,
                test,
            )

        item_bundles_base = self.get_base_bundle_rep(item_items, test)
        if self.use_item_attention and self.attention_type == "global":
            item_bundles_score = self.get_global_attention_bundle_rep(item_items, test=test)
            item_bundles_contrastive = item_bundles_score
        else:
            item_bundles_score = item_bundles_base
            item_bundles_contrastive = item_bundles_base

        return {
            "item_users": item_users,
            "item_items": item_items,
            "bundle_users": bundle_users,
            "bundle_bundles": bundle_bundles,
            "item_bundles_score": item_bundles_score,
            "item_bundles_contrastive": item_bundles_contrastive,
        }

    def gather_bundle_items(self, bundle_ids):
        flat_bundle_ids = bundle_ids.reshape(-1)
        item_ids = self.bundle_item_ids[flat_bundle_ids]
        item_mask = self.bundle_item_mask[flat_bundle_ids]
        return item_ids, item_mask

    def get_uniform_attention_bundle_rep(self, bundle_ids, item_features):
        item_ids, item_mask = self.gather_bundle_items(bundle_ids)
        item_features = item_features[item_ids]
        attention = item_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        bundle_rep = torch.sum(attention.unsqueeze(-1) * item_features, dim=-2)
        bundle_rep = bundle_rep.view(*bundle_ids.shape, self.embedding_size)
        attention = attention.view(*bundle_ids.shape, -1)
        item_ids = item_ids.view(*bundle_ids.shape, -1)
        item_mask = item_mask.view(*bundle_ids.shape, -1)
        return bundle_rep, attention, item_ids, item_mask

    def get_global_attention_bundle_rep(self, item_features, bundle_ids=None, test=False, return_details=False):
        if bundle_ids is None:
            item_ids = self.bundle_item_ids
            item_mask = self.bundle_item_mask
            target_shape = (self.num_bundles,)
        else:
            item_ids, item_mask = self.gather_bundle_items(bundle_ids)
            target_shape = bundle_ids.shape

        item_repr = item_features[item_ids]
        logits = self.global_attention_proj(item_repr).squeeze(-1)
        logits = logits.masked_fill(~item_mask, -1e9)
        attention = torch.softmax(logits, dim=-1)
        bundle_rep = torch.sum(attention.unsqueeze(-1) * item_repr, dim=-2)

        if not test:
            bundle_rep = self.attention_dropout(bundle_rep)

        if bundle_ids is not None:
            bundle_rep = bundle_rep.view(*target_shape, self.embedding_size)
            attention = attention.view(*target_shape, -1)
            item_ids = item_ids.view(*target_shape, -1)
            item_mask = item_mask.view(*target_shape, -1)

        if return_details:
            return bundle_rep, attention, item_ids, item_mask
        return bundle_rep

    def score_user_item_pairs(self, user_repr, item_repr):
        if self.attention_score_type == "dot":
            return torch.sum(user_repr.unsqueeze(-2) * item_repr, dim=-1)

        user_expand = user_repr.unsqueeze(-2).expand_as(item_repr)
        attention_input = torch.cat((user_expand, item_repr, user_expand * item_repr), dim=-1)
        return self.user_attention_mlp(attention_input).squeeze(-1)

    def get_user_attention_bundle_rep(self, user_repr, bundle_ids, item_features, test=False, return_details=False):
        item_ids, item_mask = self.gather_bundle_items(bundle_ids)
        item_repr = item_features[item_ids]
        user_flat = user_repr.reshape(-1, self.embedding_size)
        logits = self.score_user_item_pairs(user_flat, item_repr)
        logits = logits.masked_fill(~item_mask, -1e9)
        attention = torch.softmax(logits, dim=-1)
        bundle_rep = torch.sum(attention.unsqueeze(-1) * item_repr, dim=-2)

        if not test:
            bundle_rep = self.attention_dropout(bundle_rep)

        bundle_rep = bundle_rep.view(*bundle_ids.shape, self.embedding_size)
        attention = attention.view(*bundle_ids.shape, -1)
        item_ids = item_ids.view(*bundle_ids.shape, -1)
        item_mask = item_mask.view(*bundle_ids.shape, -1)

        if return_details:
            return bundle_rep, attention, item_ids, item_mask
        return bundle_rep

    def get_user_attention_bundle_rep_chunk(self, user_repr, bundle_ids, item_features):
        item_ids = self.bundle_item_ids[bundle_ids]
        item_mask = self.bundle_item_mask[bundle_ids]
        item_repr = item_features[item_ids]

        if self.attention_score_type == "dot":
            logits = torch.einsum("ud,bid->ubi", user_repr, item_repr)
        else:
            user_expand = user_repr[:, None, None, :].expand(-1, item_repr.shape[0], item_repr.shape[1], -1)
            item_expand = item_repr[None, :, :, :].expand(user_repr.shape[0], -1, -1, -1)
            attention_input = torch.cat((user_expand, item_expand, user_expand * item_expand), dim=-1)
            logits = self.user_attention_mlp(attention_input).squeeze(-1)

        logits = logits.masked_fill(~item_mask.unsqueeze(0), -1e9)
        attention = torch.softmax(logits, dim=-1)
        bundle_rep = torch.einsum("ubi,bid->ubd", attention, item_repr)
        return bundle_rep, attention, item_ids, item_mask

    def get_fusion_weight(self, item_user_repr, bundle_user_repr):
        if not self.use_view_fusion:
            return item_user_repr.new_full((item_user_repr.shape[0], 1), 0.5)

        if self.fusion_type == "global":
            beta = torch.sigmoid(self.global_fusion_logit)
            return beta.expand(item_user_repr.shape[0], 1)

        gate_input = torch.cat((item_user_repr, bundle_user_repr), dim=-1)
        return torch.sigmoid(self.user_fusion_gate(gate_input))

    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), dim=1)
        return -torch.mean(torch.log(pos_score / ttl_score))

    def compute_batch_scores(self, propagate_result, users, bundles, test=False, return_details=False):
        users = users.view(-1)
        item_user_repr = propagate_result["item_users"][users]
        bundle_user_repr = propagate_result["bundle_users"][users]
        bundle_repr = propagate_result["bundle_bundles"][bundles]

        if self.use_item_attention and self.attention_type == "user":
            user_pair_repr = item_user_repr.unsqueeze(1).expand(-1, bundles.shape[1], -1)
            item_bundle_repr, attention, attention_item_ids, attention_mask = self.get_user_attention_bundle_rep(
                user_pair_repr,
                bundles,
                propagate_result["item_items"],
                test=test,
                return_details=True,
            )
        elif self.use_item_attention and self.attention_type == "global":
            item_bundle_repr = propagate_result["item_bundles_score"][bundles]
            attention = attention_item_ids = attention_mask = None
            if return_details:
                _, attention, attention_item_ids, attention_mask = self.get_global_attention_bundle_rep(
                    propagate_result["item_items"],
                    bundle_ids=bundles,
                    test=test,
                    return_details=True,
                )
        else:
            item_bundle_repr = propagate_result["item_bundles_score"][bundles]
            attention = attention_item_ids = attention_mask = None
            if return_details:
                _, attention, attention_item_ids, attention_mask = self.get_uniform_attention_bundle_rep(
                    bundles,
                    propagate_result["item_items"],
                )

        item_bundle_repr_cl = propagate_result["item_bundles_contrastive"][bundles]
        item_scores = torch.sum(item_user_repr.unsqueeze(1) * item_bundle_repr, dim=-1)
        bundle_scores = torch.sum(bundle_user_repr.unsqueeze(1) * bundle_repr, dim=-1)
        beta = self.get_fusion_weight(item_user_repr, bundle_user_repr)
        total_scores = beta * item_scores + (1 - beta) * bundle_scores

        output = {
            "item_user_repr": item_user_repr,
            "bundle_user_repr": bundle_user_repr,
            "item_bundle_repr_score": item_bundle_repr,
            "item_bundle_repr_cl": item_bundle_repr_cl,
            "bundle_repr": bundle_repr,
            "item_scores": item_scores,
            "bundle_scores": bundle_scores,
            "beta": beta,
            "pred": total_scores,
        }

        if return_details:
            output.update(
                {
                    "attention": attention,
                    "attention_item_ids": attention_item_ids,
                    "attention_mask": attention_mask,
                }
            )

        return output

    def cal_loss(self, batch_scores):
        pred = batch_scores["pred"]
        bpr_loss = cal_bpr_loss(pred)

        item_user_repr = batch_scores["item_user_repr"].unsqueeze(1)
        bundle_user_repr = batch_scores["bundle_user_repr"].unsqueeze(1)
        item_bundle_repr_cl = batch_scores["item_bundle_repr_cl"]
        bundle_repr = batch_scores["bundle_repr"]

        user_c_loss = self.cal_c_loss(item_user_repr, bundle_user_repr)
        bundle_c_loss = self.cal_c_loss(item_bundle_repr_cl, bundle_repr)
        c_loss = (user_c_loss + bundle_c_loss) / 2
        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        users, bundles = batch
        propagate_result = self.propagate(test=False)
        batch_scores = self.compute_batch_scores(propagate_result, users, bundles, test=False)
        return self.cal_loss(batch_scores)

    def evaluate(self, propagate_result, users):
        item_user_repr = propagate_result["item_users"][users]
        bundle_user_repr = propagate_result["bundle_users"][users]

        bundle_scores = torch.mm(bundle_user_repr, propagate_result["bundle_bundles"].t())
        if self.use_item_attention and self.attention_type == "user":
            item_scores = []
            for start in range(0, self.num_bundles, self.eval_bundle_chunk_size):
                end = min(start + self.eval_bundle_chunk_size, self.num_bundles)
                chunk_bundle_ids = torch.arange(start, end, device=self.device, dtype=torch.long)
                chunk_bundle_repr, _, _, _ = self.get_user_attention_bundle_rep_chunk(
                    item_user_repr,
                    chunk_bundle_ids,
                    propagate_result["item_items"],
                )
                chunk_scores = torch.einsum("ud,ubd->ub", item_user_repr, chunk_bundle_repr)
                item_scores.append(chunk_scores)
            item_scores = torch.cat(item_scores, dim=1)
        else:
            item_scores = torch.mm(item_user_repr, propagate_result["item_bundles_score"].t())

        beta = self.get_fusion_weight(item_user_repr, bundle_user_repr)
        return beta * item_scores + (1 - beta) * bundle_scores

    def get_gate_statistics(self, propagate_result):
        beta = self.get_fusion_weight(propagate_result["item_users"], propagate_result["bundle_users"]).squeeze(-1)
        return {
            "mean_beta": beta.mean().item(),
            "std_beta": beta.std(unbiased=False).item(),
            "min_beta": beta.min().item(),
            "max_beta": beta.max().item(),
        }

    def explain_recommendation(self, propagate_result, user_ids, bundle_ids, topk=None):
        if topk is None:
            topk = self.explain_top_items

        user_ids = user_ids.view(-1)
        bundle_ids = bundle_ids.view(-1)
        item_user_repr = propagate_result["item_users"][user_ids]
        bundle_user_repr = propagate_result["bundle_users"][user_ids]
        bundle_repr = propagate_result["bundle_bundles"][bundle_ids]

        if self.use_item_attention and self.attention_type == "user":
            item_bundle_repr, attention, attention_item_ids, attention_mask = self.get_user_attention_bundle_rep(
                item_user_repr,
                bundle_ids,
                propagate_result["item_items"],
                test=True,
                return_details=True,
            )
        elif self.use_item_attention and self.attention_type == "global":
            item_bundle_repr, attention, attention_item_ids, attention_mask = self.get_global_attention_bundle_rep(
                propagate_result["item_items"],
                bundle_ids=bundle_ids,
                test=True,
                return_details=True,
            )
        else:
            item_bundle_repr, attention, attention_item_ids, attention_mask = self.get_uniform_attention_bundle_rep(
                bundle_ids,
                propagate_result["item_items"],
            )

        item_scores = torch.sum(item_user_repr * item_bundle_repr, dim=-1)
        bundle_scores = torch.sum(bundle_user_repr * bundle_repr, dim=-1)
        beta = self.get_fusion_weight(item_user_repr, bundle_user_repr).squeeze(-1)
        total_scores = beta * item_scores + (1 - beta) * bundle_scores

        valid_attention = attention.masked_fill(~attention_mask, -1.0)
        topk = min(topk, valid_attention.shape[-1])
        top_weights, top_indices = torch.topk(valid_attention, topk, dim=-1)
        top_item_ids = torch.gather(attention_item_ids, 1, top_indices)

        return {
            "score": total_scores,
            "item_view_score": item_scores,
            "bundle_view_score": bundle_scores,
            "fusion_weight": beta,
            "top_item_ids": top_item_ids,
            "top_item_weights": top_weights,
            "attention_item_ids": attention_item_ids,
            "attention_weights": attention,
        }
