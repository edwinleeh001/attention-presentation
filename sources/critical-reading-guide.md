# Transformer — key concepts and critical reading guide

Primary source: [Vaswani et al., Attention Is All You Need, arXiv v7](https://arxiv.org/pdf/1706.03762v7).

## 1. Read in this order / 阅读路线

1. Figure 1 and §3.1: identify encoder, decoder, residual order and cross-attention. / 定位编码器、解码器、残差顺序和交叉注意力。
2. Equation 1: track matrix dimensions, row normalization and weighted values. / 跟踪维度、逐行归一化与值加权。
3. §§3.2.2–3.5: connect heads, masks, FFNs and positions. / 连接多头、掩码、前馈层和位置。
4. Tables 2–3: separate test results from development ablations. / 分清测试结果和开发集消融。
5. Table 1 and Table 4: inspect computational trade-offs and architectural reuse. / 检查计算代价与架构复用。

## 2. A hand-checkable attention row / 一行注意力手算

Synthetic inputs: X = [[1,0],[0,1],[1,1]]. WQ=WK=I; WV=[[1,2],[3,0]]. Then Q=K=X and V=[[1,2],[3,0],[4,2]]. These are chosen examples, not trained weights. / 以上是人为设定的教学矩阵。

For query B: q=[0,1], dot products=[0,1,1]. With d_k=2, scaled logits=[0,1/√2,1/√2]. The causal mask gives [0,1/√2,−∞]. Let e=exp(1/√2): weights=[1/(1+e), e/(1+e), 0]. The output is [a+3b,2a], where a=1/(1+e), b=e/(1+e). / 查询 B 经因果掩码后，位置 C 权重为零；最终输出是值向量加权和。

Changing V alone with Q,K fixed changes the output but not the weights. Scaling is not unit normalization. Attention weights over positions are not next-word probabilities. / 固定 Q、K 时，改变 V 只改变输出，不改变权重；缩放不是单位归一化；位置权重不是下一词概率。

## 3. Why √d_k? / 缩放的由来

Under the paper’s independent zero-mean unit-variance coordinate assumptions, each q_i k_i has variance 1 and the sum has variance d_k. Its standard deviation is √d_k. Dividing by that factor controls scale. The assumptions motivate the operation; they do not prove learned vectors always satisfy the distribution. / 这是给定假设下的方差分析，并不表示训练后的向量必然符合该分布。

## 4. Why positions? / 为什么需要位置

Let P be a permutation matrix. With no position signal and no mask, Q′=PQ, K′=PK, V′=PV. Row softmax obeys softmax(P S Pᵀ)=P softmax(S) Pᵀ, so Attention(Q′,K′,V′)=P Attention(Q,K,V). This is permutation equivariance, not invariance. / 输出随输入重排，这是置换等变性，不是置换不变性。

## 5. Masking and generation / 掩码与生成

Shift target inputs right, then block future target positions before softmax. The diagonal is allowed because the current input is the previous target token. Training can parallelize known prefixes; inference must create unknown tokens successively. Encoder source tokens are already observed. / 右移与掩码共同防止答案泄露；训练的已知前缀可并行，未知词元的生成仍顺序进行。

## 6. Evidence audit / 实验证据核对

- Table 2 uses newstest2014; Table 3 uses EN–DE newstest2013 with no checkpoint averaging. / 不混用测试与开发结果。
- EN–FR discrepancy: abstract and Table 2 say 41.8; §6.1 prose says 41.0. This guide retains 41.8 as the table entry and does not invent a correction. / 保留文献内部差异。
- Transformer base EN–FR=38.1 does not beat every listed baseline. / 基础模型并非各列都最佳。
- Training FLOPs are estimated; Transformer cost cells span both language columns. / 成本是估算，且原表跨列显示。
- Head counts 1,4,8,16,32 score 24.9,25.5,25.8,25.8,25.4 dev BLEU; more is not monotonically better. / 更多头不单调更好。
- No label smoothing gives PPL 4.67 and BLEU 25.3, versus base 4.92 and 25.8. / 困惑度变好不必然带来 BLEU 提升。
- Parsing trains separate models. Architecture reuse is not frozen-weight zero-shot transfer. / 架构复用不等于零样本权重迁移。

## 7. Complexity / 复杂度

The attention-mixing term is O(n²d); projections add O(nd²), and FFNs O(nd·d_ff). A materialized float32 score tensor for 8 heads and n=1024 has 8×1024²×4 bytes = 32 MiB per sample per layer. This is not total training memory. Doubling n quadruples the score storage. / 32 MiB 只计算分数矩阵，不是完整训练显存。

## 8. Course relevance and follow-up / 课程关系与延伸

The original paper studies supervised translation and parsing. BERT supplies subsequent transfer-learning evidence, GPT-3 supplies in-context few-shot evidence, and ViT demonstrates visual applications. These should be separately attributed. The 2017 paper’s direct Area 2 fit is weaker than a paper centered on transfer/zero/few-shot learning; teacher approval has not been established here. / 应将后续任务证据归于后续论文，不能把基础架构直接称为零样本算法。

## 9. Suggested critical experiment / 建议批判性实验

Hold tokenization, data, training budget, parameter count and decoding fixed. Run multiple seeds, and measure quality, wall time and peak memory across sequence lengths. To test head interpretations, intervene on head outputs and evaluate task changes rather than relying solely on heatmaps. These are proposed experiments, not reported paper results. / 这些是建议实验，而非作者已做实验。

## 10. Core glossary / 核心术语

### Sequence transduction / 序列转换

Mapping an input sequence to an output sequence, such as translation.

将输入序列映射为输出序列，例如翻译。

### Autoregressive / 自回归

Each output is conditioned on previous outputs.

当前输出依赖先前输出。

### Query / 查询

A vector compared with keys to determine routing weights.

与键比较以确定取信息的权重。

### Key / 键

A vector used to score compatibility with a query.

供查询进行匹配的向量。

### Value / 值

The content vector mixed using attention weights.

被注意力权重加权混合的内容向量。

### Logit / 未归一化分数

A real-valued score before softmax.

Softmax 之前的实数分数。

### Softmax / Softmax 归一化

Exponential normalization; applied per query row in attention.

指数归一化；本文注意力按查询行操作。

### Scaled dot product / 缩放点积

A query–key dot product divided by √d_k.

将查询与键的点积除以 √d_k。

### Self-attention / 自注意力

Queries, keys and values derived from one sequence.

查询、键和值来自同一序列的表示。

### Cross-attention / 交叉注意力

Queries and key–value pairs derived from different sequences.

查询与键、值来自不同序列。

### Multi-head attention / 多头注意力

Parallel projected attention heads, concatenated and projected.

使用多组投影并行计算，拼接后再投影。

### Causal mask / 因果掩码

A mask that blocks attention to future positions.

禁止当前位置读取未来位置。

### Teacher forcing / 教师强制

Using known reference target prefixes during training.

训练中使用真实历史目标作为输入。

### Positional encoding / 位置编码

A position-dependent signal added to representations.

向表示注入序列位置的信息。

### Feed-forward network / 前馈网络

A two-layer nonlinear feature transform applied per position.

逐位置应用的两层非线性特征变换。

### Residual connection / 残差连接

An identity path that adds the input to a sublayer output.

将输入与子层输出相加的恒等路径。

### Post-norm / 后归一化

Layer normalization after residual addition.

残差相加之后进行层归一化。

### BLEU / BLEU 指标

A corpus n-gram overlap metric with brevity penalty, not accuracy.

带长度惩罚的语料级 n-gram 重合指标，不是准确率。

### Perplexity / 困惑度

Exponentiated average negative log-likelihood; tokenization matters.

由平均负对数似然指数化得到；受分词粒度影响。

### Checkpoint averaging / 检查点平均

Averaging parameters across checkpoints into one model.

将多个训练检查点的参数平均为一个模型。

## Source links

- [Original paper](https://arxiv.org/pdf/1706.03762v7)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [ViT](https://arxiv.org/abs/2010.11929)