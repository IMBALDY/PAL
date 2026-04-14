# PAL: Playtime-aware Attentive Learning for Game Bundle Recommendation

## 1. 目标

本项目的最终模型命名为 PAL，面向 Steam bundle recommendation。它从双视角 bundle 推荐基线出发，在固定 `Steam` 数据设定的前提下，逐步加入模型结构上的创新。老师关注“创新性”和“可解释性”，因此方案不能只是把 attention 直接堆到模型上，而应该回答三个问题：

- 为什么这个改动贴合 Steam bundle 数据和任务？
- 这个改动可能带来什么收益，也可能带来什么偏差或风险？
- 如何一步一步验证每个改动是否真的有效？

本方案建议采用“逐步增加复杂度”的路线。每一步只引入一个主要改动，并与前一步做消融对比，避免最后无法判断效果来自哪里。

## 2. 当前 Baseline 与实验现象

当前 PAL baseline 的核心思想是双视角学习：

- **Bundle-view**：从 user-bundle 图学习用户对整体 bundle 的偏好。
- **Item-view**：从 user-item 图和 bundle-item 图学习用户对 bundle 内 item 的偏好。
- **Cross-view contrastive learning**：让两个视角中的用户表示、bundle 表示相互对齐。

当前 Steam 的预处理方式是：

```text
1. 取前 max_bundles 个 bundle。
2. 对每个用户，保留 playtime_forever > 0 的 owned games。
3. 如果用户 owned games 与某个 bundle 的 item overlap >= min_overlap，则将该 bundle 当作用户正样本。
4. 每个用户随机留 1 个 bundle 做 tune，1 个做 test，其余做 train。
```

100 epoch 的结果中，测试集排序指标持续提升：

```text
NDCG@1  : 0.1497 -> 0.3904
NDCG@10 : 0.4064 -> 0.6522
NDCG@20 : 0.4488 -> 0.6612
```

这说明模型后期主要在提升排序质量，而不仅仅是增加 Top-K 命中数量。因此后续改进应重点观察：

```text
NDCG@1, NDCG@5, NDCG@10, MRR@10, MAP@10
```

同时保留：

```text
Recall@1, Recall@5, Recall@10, Recall@20
```

## 3. 总体路线

建议按以下顺序推进：

```text
Step 0: PAL baseline
Step 1: 模型架构改进 1 - User-aware Item Attention
Step 2: 模型架构改进 2 - Adaptive Cross-view Fusion
Step 3: 可解释性输出 - Attention Case Study
```

当前主线固定使用 Step 0 的 `Steam` 数据，不再继续做数据预处理创新。原因很直接：

- 目前需要先把 PAL 的创新点集中在**模型架构本身**，而不是继续改数据定义。
- 数据预处理变化会改变图结构、样本规模和任务难度，容易和模型创新混在一起，导致实验结论不清楚。
- 对课设汇报来说，先证明“同一份数据上，模型结构改进带来了提升”，论证会更干净。

因此，后续所有主实验都遵循一个约束：

```text
固定数据：Steam
固定正样本定义：overlap >= min_overlap
固定 train / tune / test 逻辑
只修改模型结构与解释输出
```

### 执行原则

每一步只做一个主要变化，并固定其他设置：

```text
同一组 topk: [1, 2, 5, 10, 20]
同一组 epochs / batch size / learning rate
同一组 random seed
同一份 CSV 结果格式
```

每一步都应该保存一个独立实验名，例如：

```text
step0_baseline
step1_attention
step2_fusion
step3_explain
```

这样 `results/*.csv` 中可以按 `experiment_name` 直接筛选和对比。

当前进度：

```text
Step 0 已完成：已有 100 epoch 的 baseline 结果，可作为算法基线。
数据预处理创新暂时冻结：后续主线不再推进 SteamAffinity、hard split 等数据实验。
后续工作聚焦于模型架构创新与可解释性输出。
```

## 4. Step 0：PAL Baseline

### 目的

先固定当前 PAL baseline，不加任何额外方法。

### 具体要做

不改数据和模型，只跑当前 PAL baseline：

```bash
cd PAL
python train.py -g 0 -m PAL -d Steam -i step0_baseline
```

确认输出：

```text
results/Steam_PAL_step0_baseline_<settings>_results.csv
log/Steam/PAL/
checkpoints/Steam/PAL/
```

### 需要记录

记录完整指标：

```text
R@1, R@2, R@5, R@10, R@20
N@1, N@2, N@5, N@10, N@20
MAP@1/2/5/10/20
MRR@1/2/5/10/20
F1@1/2/5/10/20
```

### 作用

Baseline 用来回答：

- 当前 PAL baseline 在 Steam 上是否能学到有效排序？
- 后续数据和模型改动是否真正超过原始方法？
- 如果某个改动让指标下降，下降是来自更难的数据，还是模型本身变差？

### 进入下一步的判断

如果 baseline 已经能稳定提升 `NDCG@5/10`，说明 pipeline 正常，可以进入 Step 1。如果 baseline 波动很大，应先固定 seed、检查 CSV 是否混入多次运行，并确认数据集规模是否太小。

## 5. Step 1：User-aware Item Attention

### 当前问题

PAL baseline 的 item-view 中，bundle 表示主要由 bundle 内 item 表示聚合得到。这个过程隐含了一个假设：

```text
bundle 内每个 item 对用户推荐决策的贡献接近
```

但 Steam bundle 中，不同用户关注的 item 可能不同。例如：

```text
Bundle A = [RPG, FPS, Puzzle, Indie, Racing]
```

对用户 1 来说，RPG 和 FPS 可能是推荐理由；对用户 2 来说，Puzzle 和 Indie 可能更重要。

### 方法

为每个 user-bundle pair 计算 bundle 内 item 的 attention：

```text
alpha(u, b, i) = softmax(score(user_u, item_i))
```

bundle 的 item-view 表示变为：

```text
h_b^u = sum_i alpha(u, b, i) * h_i
```

打分时使用个性化 bundle 表示：

```text
score_item_view(u, b) = h_u^T h_b^u
```

attention score 可以从简单形式开始：

```text
score(user_u, item_i) = user_u^T item_i
```

如果简单版本有效，再尝试 MLP：

```text
score(user_u, item_i) = MLP([user_u || item_i || user_u * item_i])
```

### 具体要做

主要修改模型文件：

```text
PAL/models/PAL.py
```

建议分两个小版本实现：

```text
Step 1a: Global Item Attention
Step 1b: User-aware Item Attention
```

Step 1a 先做全局 item attention：

```text
h_b = sum_i alpha(b, i) * h_i
```

它不依赖具体用户，计算更简单。如果 Step 1a 都不稳定，先不要直接上 user-aware attention。

Step 1b 再做 user-aware attention：

```text
h_b^u = sum_i alpha(u, b, i) * h_i
```

实现上需要注意：

```text
训练时只需对 batch 内的正负 bundles 计算 attention。
测试时如果对所有 users × all bundles × items 全量计算太慢，可以先保留原 evaluate，再单独实现 top candidate rerank，或优化 batch 计算。
```

建议新增配置项：

```yaml
use_item_attention: true
attention_type: "global"  # global / user
attention_dropout: 0.1
```

训练命令：

```bash
python train.py -g 0 -m PAL -d Steam -i step1a_global_attention
python train.py -g 0 -m PAL -d Steam -i step1b_user_attention
```

### 好处

- 贴合 bundle 推荐：bundle 是 item set，用户关注点并不相同。
- 提升短 K 排序质量的可能性较高。
- attention 权重可以直接用于解释推荐原因。

### 风险与坏处

- 参数更多，小数据集容易过拟合。
- attention 权重不一定等于真实因果解释，只能说是模型内部依据。
- 如果 bundle item 很多，逐 user-bundle-item 计算 attention 会增加计算开销。

### 可能结果

- `NDCG@1/5/10` 和 `MRR@10` 可能提升。
- `Recall@20` 可能不变或下降，因为模型更强调前排排序。
- 如果小数据集上不升反降，可能需要 attention dropout 或简化 attention 打分函数。

### 验证方式

逐步比较：

```text
M0: PAL baseline
M1: PAL + Global Item Attention
M2: PAL + User-aware Item Attention
```

如果 `M2 > M1 > M0`，说明用户相关 attention 是有价值的。

如果 `M1 > M2`，说明用户相关参数可能过拟合。

### 进入下一步的判断

如果 user-aware attention 提升 `NDCG@1/5/10` 或 `MRR@10`，进入 Step 2。如果只提升 `Recall@20` 但短 K 不升，说明它可能只是扩大命中范围，没有真正改善前排排序，需要检查 attention 是否过于分散。

## 6. Step 2：Adaptive Cross-view Fusion

### 当前问题

PAL baseline 的最终分数是两个 view 直接相加：

```text
score(u,b) = score_item_view(u,b) + score_bundle_view(u,b)
```

这相当于默认 item-view 和 bundle-view 同等重要。但不同用户的数据可靠性可能不同：

- 有些用户 item 历史很丰富，item-view 更可靠。
- 有些用户 bundle 交互更多，bundle-view 更可靠。
- 在 Steam 中，user-bundle 伪标签本身来自 item overlap，item-view 可能天然更强。

### 方法

引入用户级 view fusion gate：

```text
beta_u = sigmoid(MLP(user_embedding))
```

最终分数变为：

```text
score(u,b) =
beta_u * score_item_view(u,b)
+ (1 - beta_u) * score_bundle_view(u,b)
```

也可以先做一个无参数版本：

```text
beta_u = function(num_user_items, num_user_bundles)
```

例如 item 历史越多，`beta_u` 越大。

### 具体要做

主要修改：

```text
PAL/models/PAL.py
```

先实现一个简单可控版本：

```text
score = beta * score_item_view + (1 - beta) * score_bundle_view
```

第一版可以用全局可学习参数：

```text
beta = sigmoid(w)
```

如果有效，再改成用户级 gate：

```text
beta_u = sigmoid(MLP(user_embedding))
```

建议新增配置项：

```yaml
use_view_fusion: true
fusion_type: "global"  # global / user
fusion_reg: 0.0
```

训练命令：

```bash
python train.py -g 0 -m PAL -d Steam -i step2a_global_fusion
python train.py -g 0 -m PAL -d Steam -i step2b_user_fusion
```

同时在结果里额外记录：

```text
mean_beta
std_beta
```

如果还没来得及写入 CSV，也可以先在日志中打印。

### 好处

- 这是对 PAL 双视角框架本身的自然改进，不是硬加模块。
- 能解释模型对某个用户更依赖 item history 还是 bundle history。
- 可以缓解不同用户数据稀疏性不同的问题。

### 风险与坏处

- 如果 user-bundle 标签由 item overlap 推断，模型可能过度依赖 item-view。
- gate 可能塌缩到单一 view，例如 `beta_u` 总是接近 1。
- 需要监控 `beta_u` 分布，确认它真的在不同用户之间变化。

### 可能结果

- 如果 `NDCG@5/10` 提升，说明动态融合比固定相加更合理。
- 如果 `beta_u` 全部接近 1 或 0，说明另一个 view 没有被有效利用，需要调整正则或数据构造。

### 验证方式

比较：

```text
M2: User-aware Item Attention
M3: M2 + Adaptive Cross-view Fusion
```

额外记录：

```text
mean(beta_u)
std(beta_u)
beta_u 与用户 item 历史长度的相关性
beta_u 与用户 bundle 历史长度的相关性
```

如果 `beta_u` 随用户历史变化而变化，可解释性会更强。

### 进入下一步的判断

如果 fusion 提升短 K 指标，并且 `beta` 没有完全塌缩到 0 或 1，说明双视角自适应融合有效。如果 `beta` 塌缩，应该先保留 attention 版本，不把 fusion 放入最终主模型。

## 7. Step 3：Explanation-aware Case Study

### 目的

最后一步不一定用于提升指标，而是用于课设展示和解释性分析。

### 输出内容

对于每个用户的 Top-N 推荐，保存：

```text
user_id
recommended_bundle_id
rank
score
top_attention_item_1, weight_1
top_attention_item_2, weight_2
top_attention_item_3, weight_3
user_high_playtime_items
optional_genre_tags
```

### 具体要做

新增一个解释导出脚本，例如：

```text
PAL/export_explanations.py
```

输入：

```text
best checkpoint
config
user_id_map.csv
item_id_map.csv
bundle_id_map.csv
optional: steam_games.jsonl / bundle_data.jsonl
```

输出：

```text
results/explanations_<experiment_name>.csv
```

每行保存：

```text
user_id, recommended_bundle_id, rank, score,
top_item_1, weight_1,
top_item_2, weight_2,
top_item_3, weight_3,
user_high_playtime_items,
genre_overlap
```

### 分析方式

举例：

```text
用户历史高 playtime 游戏：
- Counter-Strike
- Left 4 Dead 2
- Borderlands

模型推荐 bundle：
- FPS / Action Bundle

attention 最高的 bundle 内游戏：
- Game A: 0.41
- Game B: 0.32
- Game C: 0.10
```

解释：

```text
模型推荐该 bundle，是因为其中注意力最高的 item 与用户历史高 playtime 的 Action/FPS 游戏偏好一致。
```

### 可解释性指标

可以额外记录：

```text
Top-1 attention mass
Top-3 attention mass
Attention entropy
Attention item 与用户历史 item 的 genre overlap
```

### 风险

- attention 只能作为模型解释，不应声称是真实因果原因。
- 如果 genre/tag 缺失较多，解释案例需要筛选字段完整的样本。

### 最终展示方式

最终报告里建议展示 2 到 3 个案例，而不是大量堆表。每个案例都回答：

```text
1. 用户历史偏好是什么？
2. PAL 推荐了哪个 bundle？
3. attention 最高的 item 是哪些？
4. 这些 item 与用户历史 playtime / genre 是否一致？
```

## 8. 建议的最终实验表

主结果表：

```text
Setting                                      R@1   R@5   R@10  N@1   N@5   N@10  MAP@10  MRR@10
Step 0: PAL baseline + original data         ...
Step 1: + user-aware item attention          ...
Step 2: + adaptive cross-view fusion         ...
Step 3: + explanation case study             ...
```

注意：当前主结果表统一使用 `Steam`，不再混入新的数据预处理设置。

```text
Steam + PAL baseline
Steam + Attention
Steam + Attention + Fusion
```

消融实验表：

```text
Model Variant                       N@1   N@5   N@10  MRR@10
PAL baseline                        ...
Global Item Attention               ...
User-aware Item Attention           ...
User-aware Attention + Fusion        ...
```

解释性案例表：

```text
User | Recommended Bundle | Top Attention Items | Attention Weights | User History Evidence
```

## 9. 最终推荐版本

如果时间有限，建议最终实现顺序是：

```text
1. 保留当前 baseline 和 CSV 指标记录。
2. 固定 Step 0 的 Steam 数据，不再继续改预处理。
3. 实现 user-aware item attention。
4. 实现 adaptive cross-view fusion。
5. 最后输出 attention case study。
```

最适合作为最终方法名的版本：

```text
PAL: Playtime-aware Attentive Learning for Game Bundle Recommendation
```

或者更短：

```text
PAL
```

它包含三个清晰创新点：

- **模型层 1**：User-aware item attention，为每个用户动态聚合 bundle 内 item。
- **模型层 2**：Adaptive cross-view fusion，动态融合 item-view 与 bundle-view。
- **解释层**：Attention case study，把推荐 bundle 中权重最高的 item 作为解释证据。

这三个点是循序渐进的：先固定 Step 0 数据和 baseline，再让 bundle 表示更个性化，最后让双视角融合更自适应，并通过 attention case study 给出可解释性证据。整个主线只在**同一份数据**上比较模型结构，方便做 ablation，也更容易说明创新点来自架构而不是数据改造。
