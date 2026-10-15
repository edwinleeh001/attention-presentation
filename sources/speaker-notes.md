# Attention Is All You Need — bilingual presentation notes

Vaswani et al., NIPS 2017. Based on arXiv:1706.03762v7. Prepared for Li Hu; 30-minute talk + 10-minute Q&A. Default slides: English; notes: Chinese, independently switchable.

These are authored teaching notes, not a verbatim translation of the paper. Numerical demos use declared synthetic matrices. No translation model was trained or benchmark reproduced. The original paper is not a direct zero/few-shot experiment; Area 2 acceptance is not established by this handout.

## 01. Can attention replace recurrence? / 注意力能替代循环结构吗？

1.5 min · [§§1–3, pp.2–3](https://arxiv.org/pdf/1706.03762v7#page=2)

### Slide points

- Task: conditional sequence modeling, p(y | x) = ∏ p(yᵢ | y<ᵢ, x). / 任务：条件序列建模，p(y | x) = ∏ p(yᵢ | y<ᵢ, x)。
- Claim: attention-based encoder–decoder representations can support strong translation without sequence-aligned recurrence or convolution. / 主张：不使用沿序列递推的循环或卷积，也能用注意力编码器—解码器取得较强翻译效果。
- Paper published in 2017; this handout follows the user-selected arXiv v7, revised in 2023. / 论文发表于 2017 年；讲义依据用户指定、于 2023 年修订的 arXiv v7。

### English speaker notes

Open with a translation task: a source sentence is known, but its translated sentence must be produced. The central question is not whether attention exists—earlier systems already used it. It is whether attention can become the main mechanism for exchanging information between positions. Keep the factorization visible: each output token depends on the source and earlier outputs. This matters later because removing recurrent layers does not remove autoregressive generation. Our route is to inspect the architecture, compute a small attention example, then ask what the experiments establish. We will separate the original 2017 contribution from the later transfer-learning systems that reused its components.

### 中文讲稿

先用翻译任务开场：输入句子已知，目标句子需要逐步生成。本文不是首次提出注意力，而是把注意力变成不同位置交换信息的主要机制。请指向概率分解：当前输出依赖源句和之前的输出。移除循环层，并不等于生成时不再自回归。讲解路线是整体结构、一次可计算的注意力，再到实验能支持什么结论。最后会区分 2017 年原始贡献与后续迁移学习应用。

### Audience prompt / 课堂互动

Ask: does “no recurrence” imply that all unknown output tokens can be generated simultaneously? Revisit after masking.

提问：“没有循环层”是否表示能同时生成所有未知输出词？讲完掩码后再回答。

## 02. Shorten the communication path / 缩短信息传递路径

1.5 min · [§§1–2; Table 1, pp.2,6–7](https://arxiv.org/pdf/1706.03762v7#page=2)

### Slide points

- An RNN computes a hidden state from the preceding hidden state. / RNN 当前隐藏状态依赖前一个隐藏状态。
- Convolutions can process positions in parallel, but distant positions require a growing receptive field. / 卷积可并行处理位置，但远距离交互需要扩大的感受野。
- Dense self-attention lets every position directly access all permitted positions within a layer. / 稠密自注意力让每个位置在一层内直接访问所有允许位置。

### English speaker notes

Imagine the first and last words of a long sentence exchanging information. An RNN must pass information through a chain of states. A convolutional network can parallelize positions, yet a single small kernel cannot connect the whole sequence. Attention computes a direct content-dependent connection between two positions. That short path is a structural property, not a guarantee that the model will learn the correct linguistic relation. Likewise, a constant number of sequential operations per attention layer does not imply constant computational cost: the matrix contains a score for every query–key pair. This distinction will be essential when we discuss quadratic scaling.

### 中文讲稿

想象长句首尾两个词要交换信息。RNN 需要沿状态链传递；卷积能并行，但小卷积核不能一次覆盖全句。注意力允许位置之间建立依赖内容的直接连接。短路径属于结构性质，并不保证学到正确语言关系。每层的顺序操作数为常数，也不代表计算量不随长度变化，因为仍要计算每个查询—键对。到复杂度页时，要再次区分这两个概念。

### Audience prompt / 课堂互动

Compare path length, sequential operations and total work; do not use “faster” as a substitute for all three.

请分别比较路径长度、顺序操作数和总计算量，不要用“更快”概括三者。

## 03. What was actually new? / 真正的新意是什么？

2 min · [§§2–3, pp.2–6](https://arxiv.org/pdf/1706.03762v7#page=2)

### Slide points

- Earlier work already used attention and self-attention. / 此前已有注意力和自注意力研究。
- The Transformer combines scaled dot products, learned attention heads, positional information and feed-forward blocks. / Transformer 组合了缩放点积、多组可学习注意力头、位置信息和前馈网络。
- “All you need” does not mean the network contains only an attention operator. / “All you need”不意味着网络只有注意力算子。

### English speaker notes

Avoid the tempting historical claim that this paper invented attention. Its background explicitly cites attention in translation and self-attention in other tasks. The architectural change is to replace sequence-aligned recurrence and convolution with attention as the main route for contextual interaction. The surrounding ingredients still matter: embeddings represent tokens, positional encodings expose order, feed-forward layers transform features, and residual connections and normalization support optimization. A strong presentation should explain how these parts cooperate rather than treat the title literally. We also need empirical checks: translation quality, cost estimates and controlled variations. These tests answer different questions, so we will not merge them into one claim of universal superiority.

### 中文讲稿

不要说本文发明了注意力。背景章节已经引用翻译注意力及其他自注意力研究。主要变化是用注意力承担上下文交互，替代沿序列对齐的循环与卷积。周边结构仍不可少：词嵌入表示词元，位置编码提供顺序信息，前馈层变换特征，残差与归一化帮助优化。汇报应解释这些结构如何合作，而不是按标题字面理解。翻译质量、训练成本估算和结构消融检验的是不同问题，不能合并成“全面更优”。

### Audience prompt / 课堂互动

Name two essential non-attention components before showing the framework.

在展示框架前，让听众说出两个必需的非注意力结构。

## 04. Read the full Transformer / 读懂完整 Transformer

1 min · [Figure 1; §§3.1–3.4, pp.3–5](https://arxiv.org/pdf/1706.03762v7#page=3)

### Slide points

- Base configuration: 6 encoder layers, 6 decoder layers, d_model = 512, 8 heads. / 基础配置：6 层编码器、6 层解码器，d_model = 512，8 个头。
- Encoder memory supplies the keys and values of decoder cross-attention. / 编码器记忆为解码器交叉注意力提供键和值。
- The original residual blocks are post-norm: LayerNorm(x + Sublayer(x)). / 原始残差块采用后归一化：LayerNorm(x + Sublayer(x))。

### English speaker notes

Read from the source tokens through the encoder to its memory. Then follow shifted target tokens through the decoder. The decoder has an extra attention operation: it reads the encoder memory, using its own states as queries. Emphasize the two different softmax uses. Inside an attention head, softmax normalizes over permitted key positions. At the top of the decoder, softmax normalizes over vocabulary items to predict a token. The six-layer repeats exclude the input embeddings and output classifier. The diagram deliberately depicts the 2017 post-norm architecture; modern implementations may reorder normalization or replace positional encodings, but those are not the architecture evaluated here.

### 中文讲稿

先沿源词元读到编码器记忆，再沿右移后的目标词元读完整个解码器。解码器多出的注意力使用自身状态作为查询，读取编码器记忆。强调两处 Softmax 的区别：头内部在允许的键位置上归一化；解码器顶部在词表上归一化以预测词元。六层重复不包括输入嵌入和输出分类器。这里明确画的是 2017 年后归一化结构，现代实现中的其他顺序或位置编码不能直接当作本文原设定。

### Audience prompt / 课堂互动

Point to the only attention block whose Q and K,V come from different sequences.

指出唯一一个 Q 与 K、V 来自不同序列的注意力结构。

## 05. Q asks, K matches, V carries content / Q 发问，K 匹配，V 传递内容

1 min · [Eq.1; §3.2, pp.4–5](https://arxiv.org/pdf/1706.03762v7#page=4)

### Slide points

- Q ∈ ℝⁿq×dₖ, K ∈ ℝⁿk×dₖ, V ∈ ℝⁿk×dᵥ. / Q ∈ ℝⁿq×dₖ，K ∈ ℝⁿk×dₖ，V ∈ ℝⁿk×dᵥ。
- Scores have shape n_q × n_k; the output has shape n_q × d_v. / 分数矩阵形状为 n_q × n_k；输出为 n_q × d_v。
- For self-attention, Q,K,V come from the same sequence through different learned projections. / 自注意力中，Q、K、V 来自同一序列，但经过不同可学习投影。

### English speaker notes

The query–key–value analogy is useful only if we attach it to the computation. A query compares against keys and produces one score per key position. Softmax converts that row of scores into weights. Those weights combine value vectors, not key vectors. The number of keys must equal the number of values, while the number of queries can differ. This is why the same operator handles cross-attention between source and target sentences of different lengths. In self-attention, sharing the input sequence does not mean the projected Q, K and V matrices are identical. Their learned projection matrices can be different.

### 中文讲稿

类比必须落到计算：一个查询与各键比较，为每个键位置产生分数；Softmax 将这一行分数变成权重；最后被加权的是值向量，不是键。键和值的数量必须一致，查询数量可以不同，所以同一算子可处理长度不同的源句和目标句。自注意力虽然共用输入序列，但 Q、K、V 的投影矩阵通常不同，并不意味着它们数值相同。

### Audience prompt / 课堂互动

If the source has 5 tokens and the decoder has 3 queries, what is the cross-attention weight shape? Answer: 3 × 5.

源句 5 个词元，解码器 3 个查询，交叉注意力权重是什么形状？答案：3 × 5。

## 06. Scale, normalize, aggregate / 缩放、归一化、聚合

1.5 min · [Eq.1 and footnote 4, p.4](https://arxiv.org/pdf/1706.03762v7#page=4)

### Slide points

- Under independent zero-mean, unit-variance coordinates, Var(q·k) = d_k. / 在各分量独立、均值为零、方差为一的假设下，Var(q·k) = d_k。
- Dividing by √d_k controls score scale; it does not make Q and K unit vectors. / 除以 √d_k 控制分数尺度，并不会把 Q、K 归一化成单位向量。
- Mask forbidden scores before softmax; normalized weights then sum to one over allowed positions. / 在 Softmax 之前屏蔽不允许的分数；归一化权重在允许位置上的和为一。

### English speaker notes

Walk through the formula as four operations rather than one symbol. First take dot products. Second divide by the square root of key dimension. Third normalize each row. Fourth multiply by V. Under the paper’s simplifying variance assumptions, a dot product grows in variance with dimension; scaling counteracts that effect. Large logits can make softmax very sharp, leaving small derivatives for many entries. This is a motivation, not a proof that every unscaled attention model fails. Do not confuse scaling with cosine similarity: vector lengths still matter. In the next example, all matrices are deliberately small and supplied explicitly, so every weight and output can be independently checked.

### 中文讲稿

把公式分成四步：点积、除以键维度的平方根、逐行归一化、乘 V。在论文给出的简化独立方差假设下，点积方差随维度增长，缩放抵消这个效应。分数很大时 Softmax 会很尖锐，很多项的导数变小。这是动机，不是所有不缩放模型必失败的证明。缩放也不等于余弦相似度，向量长度依然影响结果。下一页使用明确给定的小矩阵，每个权重和输出都可独立复算。

### Audience prompt / 课堂互动

Why divide by √d_k rather than d_k? Connect the answer to standard deviation.

为什么除以 √d_k 而不是 d_k？请从标准差解释。

## 07. Compute one attention row / 亲手计算一行注意力

2.5 min · [Computed teaching example based on Eq.1 and §3.2.3, pp.4–5](https://arxiv.org/pdf/1706.03762v7#page=4)

### Slide points

- Synthetic vectors and projections; no trained translation model is running. / 使用合成向量与投影，并未运行训练好的翻译模型。
- Playback reveals exact computation stages; it is not a training trajectory. / 播放逐步展示精确计算，不是训练轨迹。
- Changing parameters recomputes the output and clears previous quiz feedback. / 改变参数会重算输出，并清除先前的预测反馈。

### English speaker notes

Start with the middle query and ask the audience whether the last position receives any weight under a causal mask. Let someone choose an answer before revealing it. Step through the dot products, scale, mask and row softmax. The masked entry is negative infinity before normalization and exactly zero afterward. Now turn the mask off without changing the underlying vectors. The future position contributes again. Switch off scaling and compare the concentration of the weight distribution. Finally inspect the weighted sum: it is a mixture of value vectors, so it need not equal any one value. The example isolates a single head and excludes dropout, residuals, normalization and the output vocabulary classifier.

### 中文讲稿

从中间查询开始，先让听众预测因果掩码下最后一个位置有没有权重。再逐步显示点积、缩放、掩码和逐行 Softmax。被遮挡分数在归一化前是负无穷，之后权重精确为零。保持向量不变，关闭掩码，未来位置重新参与。关闭缩放，比较权重分布的集中程度。最后看加权和，它是值向量的混合，不必等于某一个值。本例隔离单个头，不含 Dropout、残差、归一化或词表输出层。

### Audience prompt / 课堂互动

Predict before pressing Play: can a future position receive nonzero weight under the causal mask?

播放前预测：因果掩码下，未来位置能获得非零权重吗？

## 08. Different projections, different views / 不同投影，不同关注方式

2 min · [§3.2.2, pp.4–5; Table 3(A), p.9](https://arxiv.org/pdf/1706.03762v7#page=5)

### Slide points

- head_i = Attention(QWᵢQ, KWᵢK, VWᵢV). / head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)。
- MultiHead = Concat(head₁,…,head_h)Wᴼ, rather than an unweighted average. / MultiHead = Concat(head₁,…,head_h)Wᴼ，并非简单等权平均。
- Base: h=8 and d_k=d_v=64, so 8×64=512. / 基础模型 h=8，d_k=d_v=64，因此 8×64=512。

### English speaker notes

A common misconception is that multi-head attention copies the same attention map eight times. The projections are head-specific, so each head can compare positions in a different feature space. The interactive two-head example uses hand-specified projections: one reads the first coordinate and another the second. Toggle identical projections to see the special case in which the maps coincide. These heads are not learned linguistic specialists; the chosen matrices simply make the mechanism inspectable. After attention, concatenate their outputs and apply a learned output projection. With the paper’s width allocation, head dimensions shrink as the head count increases. More heads are therefore not automatically more capacity or better results; the ablation will test this.

### 中文讲稿

多头不是把同一张注意力图复制八份。各头投影不同，因此能在不同特征空间比较位置。交互示例的两个头使用人为指定的投影，分别读取第一和第二个坐标；切换成相同投影后，可以看到注意力图重合的特殊情况。它们不是训练得到的语言学专家，只用于展示机制。各头输出先拼接，再经过输出投影。论文按头数分配总宽度，头越多每头越窄，所以更多头不自动意味着更强，后面的消融会检验这一点。

### Audience prompt / 课堂互动

Toggle identical projections: is diversity guaranteed by the name “multi-head”?

切换为相同投影：只叫“多头”就保证多样性吗？

## 09. Three attention patterns / 三种注意力连接方式

1.5 min · [§§3.1, 3.2.3, pp.3,5](https://arxiv.org/pdf/1706.03762v7#page=5)

### Slide points

- Encoder self-attention: Q,K,V from source-side states; all source positions allowed. / 编码器自注意力：Q、K、V 来自源序列状态，可访问全部源位置。
- Decoder self-attention: target-side Q,K,V with a causal mask. / 解码器自注意力：Q、K、V 来自目标侧，并使用因果掩码。
- Cross-attention: decoder Q, encoder-memory K,V; all non-padding source positions allowed. / 交叉注意力：解码器提供 Q，编码器记忆提供 K、V；可访问所有非填充源位置。

### English speaker notes

Use a three-token target example. To predict the target sequence, training inputs are shifted right: a start symbol followed by the earlier reference tokens. The causal mask prevents a position from reading later target inputs. Its diagonal is allowed because that position contains the previous target token, not the answer it is predicting. Training can compute all these known input positions in parallel. During generation, later target tokens are not known yet, so the decoder still produces tokens successively. Source tokens are already given, so cross-attention does not require a causal source mask. Padding masks are a separate issue: padding is not a real token and should not receive attention.

### 中文讲稿

以三个目标词元为例，训练输入右移：起始符后接更早的真实目标词。因果掩码阻止当前位置读到更后面的目标输入。对角线可以保留，因为右移后当前位置装的是之前的目标词，而不是当前要预测的答案。训练时这些输入已知，可并行计算；生成时后续词未知，仍需逐步产生。源句已给定，交叉注意力不需对源位置施加因果限制。填充掩码是另一个问题，Padding 不是实际词元。

### Audience prompt / 课堂互动

Why is the diagonal allowed, while future target positions are forbidden?

为什么允许对角线，却不允许未来目标位置？

## 10. Give tokens a sense of order / 让词元获得顺序信息

1 min · [§3.5, p.6; Table 3(E), p.9](https://arxiv.org/pdf/1706.03762v7#page=6)

### Slide points

- Add sinusoidal positional vectors to scaled token embeddings. / 将正弦位置向量加到缩放后的词嵌入上。
- PE(pos,2i)=sin(pos/10000^(2i/d)); PE(pos,2i+1)=cos(pos/10000^(2i/d)). / PE(pos,2i)=sin(pos/10000^(2i/d))；PE(pos,2i+1)=cos(pos/10000^(2i/d))。
- Learned positional embeddings gave nearly the same dev result; extrapolation was a motivation, not a demonstrated guarantee. / 可学习位置嵌入的开发集结果很接近；外推是设计动机，不是已证明的保证。

### English speaker notes

If we permute a sequence of content vectors and apply unmasked self-attention without positional information, the outputs permute accordingly. There is no separate signal saying which word came first. Positional encoding breaks that ambiguity by adding a position-dependent vector. The sine and cosine coordinates use different frequencies, giving many timescales. The formula can be evaluated at unseen positions, but being defined there is not the same as reliable long-context generalization. Table 3 compares learned positions against sinusoids and finds similar development performance. Therefore do not claim that sinusoidal positions are the only possible choice or that the paper proved indefinite length extrapolation.

### 中文讲稿

若对不含位置信息的内容向量重新排列，无掩码自注意力的输出也相应重排，它没有额外信息知道哪个词在前。位置编码通过加入依赖位置的向量消除这种歧义。正弦与余弦使用不同频率，覆盖多种尺度。公式可以在未见位置计算，但“有定义”不等于长上下文泛化可靠。表 3 的可学习位置与正弦位置开发集表现接近，因此不能说正弦位置是唯一选择，也不能说论文证明了无限长度外推。

### Audience prompt / 课堂互动

Distinguish “the formula is defined for a new position” from “the model works well there.”

区分“新位置上公式有定义”和“模型在那里表现良好”。

## 11. Attention mixes positions; FFNs mix features / 注意力混合位置，FFN 变换特征

1.5 min · [§§3.1,3.3,5.4, pp.3,5,8](https://arxiv.org/pdf/1706.03762v7#page=5)

### Slide points

- FFN(x)=max(0,xW₁+b₁)W₂+b₂; 512 → 2048 → 512 in the base model. / FFN(x)=max(0,xW₁+b₁)W₂+b₂；基础模型维度为 512 → 2048 → 512。
- FFN weights are shared across positions within a layer, not across all layers. / 同一层的 FFN 在各位置共享参数，并非所有层共享参数。
- Original order: sublayer → dropout → residual addition → layer normalization. / 原始顺序：子层 → Dropout → 残差相加 → 层归一化。

### English speaker notes

Once attention has collected information from other positions, the position-wise feed-forward network applies a nonlinear feature transformation to each position. It uses the same transformation at all positions of that layer, while a different layer has its own parameters. The output width returns to 512 so that residual addition is dimensionally valid. In the original architecture, dropout is applied to the sublayer output before adding the residual and applying normalization. Show why the residual is not another attention weight: it is a separate identity path. These details prevent us from silently replacing the paper’s architecture with a modern pre-norm variant while claiming to explain the original experiment.

### 中文讲稿

注意力从其他位置收集信息后，逐位置 FFN 对每个位置进行非线性特征变换。同一层各位置使用相同变换，不同层有不同参数。输出回到 512 维，才能与残差相加。原始结构先对各子层输出做 Dropout，再加残差，最后归一化。残差不是另一个注意力权重，而是独立的恒等路径。讲清这些顺序，可以避免把现代前归一化变体悄悄替换成论文结构。

### Audience prompt / 课堂互动

Can the FFN alone mix information across tokens? No: it acts independently at each position.

单独的 FFN 能混合不同词元信息吗？不能，它逐位置独立作用。

## 12. Training is part of the method / 训练设置也是方法的一环

1 min · [§§5.2–5.4,6.1–6.2, pp.7–9](https://arxiv.org/pdf/1706.03762v7#page=7)

### Slide points

- Adam: β₁=.9, β₂=.98, ε=10⁻⁹; warmup for 4,000 steps, then inverse-square-root decay. / Adam：β₁=.9、β₂=.98、ε=10⁻⁹；预热 4,000 步，然后按平方根倒数衰减。
- Base: dropout .1, label smoothing .1, 100K steps; big: 300K steps. / 基础模型：Dropout .1、标签平滑 .1、100K 步；大模型：300K 步。
- Training targets are known; generation uses beam search with predicted prefixes. / 训练时目标已知；生成时使用预测前缀进行束搜索。

### English speaker notes

The learning-rate schedule first rises linearly and then decays. It is not a constant-rate Adam run. Regularization includes dropout and label smoothing. The paper notes that smoothing can worsen perplexity while improving BLEU, a useful reminder that optimizing a probability metric and improving task output are not identical. Base and big models also differ in size and training duration, so their score difference cannot isolate head count alone. For final translation evaluation, checkpoint averaging and beam search belong to the inference protocol. We will distinguish that protocol from the development-set ablations, which do not use checkpoint averaging.

### 中文讲稿

学习率先线性增大再衰减，并非恒定学习率的 Adam。正则化包括 Dropout 与标签平滑。论文指出，平滑可能让困惑度变差，却改善 BLEU，说明概率指标和任务质量不完全一致。基础与大模型同时改变规模和训练时长，其分数差不能单独归因于头数。最终翻译结果还包含检查点平均与束搜索，后面的开发集消融却没有检查点平均，要分清评估协议。

### Audience prompt / 课堂互动

Would lower perplexity always imply higher BLEU? Use label smoothing as a counterexample.

更低困惑度是否必然带来更高 BLEU？用标签平滑作为反例。

## 13. Know the dataset and the metric / 先弄清数据集与指标

1.5 min · [§5.1; Tables 2–4, pp.7–10](https://arxiv.org/pdf/1706.03762v7#page=7)

### Slide points

- WMT14 EN–DE: about 4.5M sentence pairs, shared ~37K BPE vocabulary. / WMT14 英德：约 450 万句对，共用约 3.7 万 BPE 词表。
- WMT14 EN–FR: 36M sentence pairs, 32K word-piece vocabulary. / WMT14 英法：3,600 万句对，3.2 万词片词表。
- Table 2: newstest2014; Table 3: EN–DE newstest2013 dev; Table 4: WSJ section 23. / 表 2：newstest2014；表 3：英德 newstest2013 开发集；表 4：WSJ 第 23 节。

### English speaker notes

Before comparing any numbers, identify the task, split and metric. BLEU measures corpus-level overlap with reference translations through modified n-gram precision and a brevity penalty; it is not a percentage of sentences translated correctly. The paper’s main comparison is on newstest2014, while the ablations use newstest2013. Parsing uses constituent F1, so those scores are not comparable to BLEU. Training-cost figures are estimates from runtime, GPU counts and assumed throughput, not measurements from a single controlled hardware benchmark. Ask what each comparison holds fixed and what differs. This is more informative than reporting the largest number in each table.

### 中文讲稿

比较数字前先确认任务、划分和指标。BLEU 结合修正的 n-gram 精确率与长度惩罚衡量语料级参考译文重合，并非正确翻译句子的百分比。主要结果在 newstest2014，消融在 newstest2013。句法分析用成分 F1，不能与 BLEU 比大小。训练成本由时长、GPU 数量及吞吐假设估算，不是统一硬件条件下的实测竞速。请说明每个比较固定了什么，又改变了什么。

### Audience prompt / 课堂互动

Why must 25.8 dev BLEU not be placed beside 28.4 test BLEU as a controlled comparison?

为什么不能把 25.8 开发集 BLEU 与 28.4 测试集 BLEU 当作受控对比？

## 14. Translation evidence, with its boundaries / 翻译证据及其边界

2 min · [Table 2 and §6.1, p.8; abstract, p.1](https://arxiv.org/pdf/1706.03762v7#page=8)

### Slide points

- Base reports 27.3 EN–DE and 38.1 EN–FR; it is not the best EN–FR entry. / 基础模型报告英德 27.3、英法 38.1，并非英法最优条目。
- 28.4 − 26.36 = 2.04 BLEU above the listed ConvS2S ensemble on EN–DE. / 英德上比表中的 ConvS2S 集成高 28.4 − 26.36 = 2.04 BLEU。
- Version discrepancy: §6.1 says 41.0 EN–FR, while Table 2 and the abstract say 41.8. We retain the table value and flag the difference. / 版本差异：§6.1 正文写英法 41.0，表 2 与摘要写 41.8。这里保留表值并标明差异。

### English speaker notes

Read the EN–DE and EN–FR columns separately. Big is strong in both, but base’s EN–FR score is below several listed baselines. Therefore the statement that every Transformer configuration wins is false. The EN–DE improvement over the best listed ensemble is 2.04 BLEU, an absolute score difference rather than a percent increase. The paper compares results collected from prior work and estimates training FLOPs, so differences in systems and training setups remain. There is also a visible inconsistency within the selected version: the EN–FR prose says 41.0 while Table 2 and the abstract say 41.8. We explicitly use 41.8 as the table entry without silently rewriting the prose or claiming a resolved correction.

### 中文讲稿

英德与英法要分开读。大模型两者都强，但基础模型英法低于多个基线，因此“所有 Transformer 配置都赢”不成立。英德相对表中最佳集成提升 2.04 BLEU，是绝对分差，不是百分比提升。比较结果来自先前论文，训练 FLOPs 也属于估算，系统与训练条件差异仍存在。当前版本内部还存在英法正文 41.0、表格和摘要 41.8 的不一致。我们明确使用表格值 41.8，同时保留差异说明，不擅自替作者修正文献。

### Audience prompt / 课堂互动

Which claim fails if you inspect only the EN–FR base row?

只检查英法基础模型这一行，就能推翻什么过度概括？

## 15. More heads is not always better / 更多注意力头不一定更好

2 min · [Table 3 and §6.2, p.9](https://arxiv.org/pdf/1706.03762v7#page=9)

### Slide points

- Head counts 1,4,8,16,32: dev BLEU 24.9,25.5,25.8,25.8,25.4. / 头数 1、4、8、16、32：开发集 BLEU 为 24.9、25.5、25.8、25.8、25.4。
- No dropout: 24.6; no label smoothing: 25.3 with lower PPL 4.67. / 无 Dropout：24.6；无标签平滑：25.3，但困惑度较低，为 4.67。
- Learned positions: 25.7 versus sinusoidal base 25.8. / 可学习位置：25.7；正弦位置基础模型：25.8。

### English speaker notes

Use the head-count selector to trace a non-monotonic pattern. One head is worse than eight here, but thirty-two is also worse. The head dimensions change to maintain the overall width, so this experiment is not simply adding extra full-width heads. Next compare no smoothing with base: perplexity improves numerically while BLEU worsens. Finally compare learned and sinusoidal positions; their development scores are close. These observations support design choices in this setting, not universal laws. Table 3 uses no checkpoint averaging, unlike Table 2. It reports point results without the multi-seed uncertainty needed to establish that every small decimal difference is reliable.

### 中文讲稿

用头数选择器观察非单调变化：一个头不如八个，但三十二个也更低。头数变化时每头维度同时变化，以保持整体宽度，并非不断添加全宽头。再看无标签平滑：困惑度变好，BLEU 却变差。可学习与正弦位置结果相近。这些支持特定设置下的设计选择，不构成普遍规律。表 3 不使用检查点平均，与表 2 不同；表中也缺少多随机种子不确定性，不能把每个小数差都说成可靠提升。

### Audience prompt / 课堂互动

If 8 and 16 heads both score 25.8, what evidence would establish a robust preference?

8 和 16 个头同为 25.8，还需要什么证据才能可靠偏好其中一个？

## 16. Parallel does not mean free / 可并行不等于没有代价

1.5 min · [Table 1 and §4, pp.6–7; memory estimate derived here](https://arxiv.org/pdf/1706.03762v7#page=6)

### Slide points

- Table 1: attention mixing O(n²d), recurrence O(nd²); Q/K/V projections add O(nd²). / 表 1：注意力混合 O(n²d)，循环 O(nd²)；Q/K/V 投影另需 O(nd²)。
- A materialized attention map stores n² weights per head. / 显式保存的注意力图，每个头含 n² 个权重。
- O(1) path length is within a dense attention layer, not the total decoding latency. / O(1) 路径长度指稠密注意力层内，不是完整解码时延。

### English speaker notes

Increase the length slider and observe the score-matrix entry count. Doubling length multiplies this count by four. The display is a formula-based memory estimate for stored float32 scores, not a GPU benchmark or a full training memory measurement. The paper’s complexity table emphasizes the attention mixing operation. A full block also includes projections and a feed-forward network, so repeating only O(n²d) as the entire Transformer cost is incomplete. Likewise, the training computation over known positions can be parallel while autoregressive output generation is still sequential. This trade-off motivates later efficient-attention work, but those implementations are outside the experiment being explained.

### 中文讲稿

增加长度滑块，观察分数矩阵元素数量。长度翻倍，数量变为四倍。这里只按 float32 显式分数矩阵计算存储估算，并非 GPU 实测或完整训练显存。论文复杂度表主要突出注意力混合，完整块还包含投影与 FFN，因此只用 O(n²d) 概括整个 Transformer 不完整。已知输入位置的训练计算可以并行，输出生成仍自回归。这一代价促使后续高效注意力研究，但那些实现不属于本文实验。

### Audience prompt / 课堂互动

At n=1,024 and 8 heads, how many float32 MiB do stored attention scores require per sample per layer? Answer: 32 MiB.

n=1,024、8 个头时，每样本每层的 float32 注意力分数占多少 MiB？答案：32 MiB。

## 17. A second task tests architectural reuse / 第二项任务检验架构复用

1 min · [§6.3 and Table 4, pp.9–10](https://arxiv.org/pdf/1706.03762v7#page=10)

### Slide points

- Four-layer Transformer: WSJ-only F1 91.3; semi-supervised F1 92.7. / 四层 Transformer：仅 WSJ 训练 F1 为 91.3；半监督为 92.7。
- Dyer et al. WSJ discriminative model: 91.7; generative model: 93.3. / Dyer 等的 WSJ 判别模型为 91.7；生成模型为 93.3。
- This does not demonstrate zero-shot transfer of a frozen translation model. / 这并不证明冻结翻译模型后的零样本迁移。

### English speaker notes

Parsing converts a sentence into a structured constituent representation. It tests whether the architectural idea can be trained for a different output structure. The Transformer is competitive, but not the strongest number in every training category. The WSJ-only result is lower than the listed discriminative Dyer model, and the generative Dyer result is higher again under a different setting. Keep those training categories visible. The authors train parsing models; they do not simply take a frozen translation checkpoint and ask it to parse without task data. Architectural generality and zero-shot transfer are different claims, which is especially important for an Area 2 discussion.

### 中文讲稿

句法分析将句子转为结构化成分表示，检验同一架构思路能否训练于另一种输出结构。Transformer 表现有竞争力，但不是所有训练类别下的最大数字。仅 WSJ 设置低于表中 Dyer 判别模型，生成模型又在不同设置下更高。必须显示训练类别。作者训练了句法分析模型，并未直接冻结翻译检查点、不用任务数据就完成解析。架构通用性与零样本迁移是不同主张，这对 Area 2 尤其重要。

### Audience prompt / 课堂互动

Is reusing an architecture equivalent to transferring pretrained weights?

复用架构是否等于迁移预训练权重？

## 18. What the paper does not establish / 论文没有证明什么

1.5 min · [§4, §§6–7 and attention visualizations, pp.7–10,13–15](https://arxiv.org/pdf/1706.03762v7#page=13)

### Slide points

- Attention visualizations suggest patterns; they do not prove causal explanations. / 注意力可视化提示模式，但不能证明因果解释。
- Machine translation and parsing results do not establish universal reasoning or few-shot ability. / 翻译与句法分析结果不能证明普遍推理或少样本能力。
- Historical cost estimates, limited task coverage and absent multi-seed intervals constrain conclusions. / 历史成本估算、有限任务覆盖及缺少多种子区间限制了结论范围。

### English speaker notes

Use the appendix carefully. A head attending to a syntactic relation is an interesting observation, but a weight map alone is not proof that the connection caused the output. A causal claim would need intervention or another suitable analysis. The paper also does not evaluate every possible sequence task or report modern language-model few-shot benchmarks. Its training-cost comparison is historically useful without being a hardware-neutral speed guarantee. An informative follow-up experiment would hold data, tokenization, training budget and decoding fixed, run several seeds, and measure quality as well as wall time and memory across lengths. Present this as our proposed evaluation, not as an experiment the authors performed.

### 中文讲稿

附录中的头似乎关注某种句法关系，这是观察证据，但注意力图本身不证明该连接导致输出，因果论断需要干预等分析。本文也没有覆盖所有序列任务，更没有现代语言模型的 few-shot 基准。训练成本比较具有历史意义，但不是不依赖硬件的速度保证。可提出后续实验：固定数据、分词、预算和解码设置，运行多个种子，并随长度测量质量、时延和显存。必须标为我们的建议，而不是作者已经完成的实验。

### Audience prompt / 课堂互动

Design one intervention that tests an attention-head interpretation.

设计一个干预实验，检验对某个注意力头的解释。

## 19. From architecture to transfer learning / 从架构走向迁移学习

1.5 min · [Original §7, p.10; Devlin et al. 2018; Brown et al. 2020; Dosovitskiy et al. 2020](https://arxiv.org/pdf/1706.03762v7#page=10)

### Slide points

- BERT: bidirectional Transformer pretraining followed by task fine-tuning. / BERT：双向 Transformer 预训练，再面向任务微调。
- GPT-3: autoregressive language modeling with few-shot demonstrations in context, without task-time gradient updates. / GPT-3：自回归语言建模，通过上下文示例进行少样本任务，不在任务时更新梯度。
- ViT: Transformer-based image recognition with large-scale pretraining and transfer. / ViT：基于 Transformer 的图像识别，结合大规模预训练与迁移。

### English speaker notes

This is the bridge to the course area, not a rewrite of the 2017 experimental claim. BERT, GPT-3 and ViT illustrate different ways to reuse Transformer ideas in language understanding, in-context task adaptation and vision. Their objectives, model layouts, data and evaluation protocols differ. For a formal Area 2 submission, the original paper is an architectural foundation; the direct topic match is weaker than a paper whose main experiment is transfer, zero-shot or few-shot learning. This handout does not claim teacher approval. If asked why it matters to Area 2, answer that it explains the machinery underlying later systems, and cite the later papers for the actual transfer evidence.

### 中文讲稿

这是连接课程主题的桥梁，不是重写 2017 年实验主张。BERT、GPT-3、ViT 分别体现语言理解迁移、上下文任务适应和视觉中的 Transformer 应用，但目标函数、结构、数据与评估不同。若作为 Area 2 正式选文，原论文属于架构基础，与“主要实验就是迁移、零样本或少样本”的论文相比，直接匹配较弱。本讲义不代表教师已批准。被问及为何相关时，应回答它解释后续系统的底层机制，并将实际迁移证据归给对应论文。

### Audience prompt / 课堂互动

Which evidence belongs to 2017, and which belongs to BERT, GPT-3 or ViT?

哪些证据属于 2017 原论文，哪些属于 BERT、GPT-3 或 ViT？

## 20. Three ideas to take away / 带走三个核心认识

1 min · [§§3–7, pp.3–10](https://arxiv.org/pdf/1706.03762v7#page=10)

### Slide points

- Mechanism: projected Q/K determine weights; V supplies the mixed content. / 机制：投影后的 Q、K 决定权重，V 提供被混合内容。
- Architecture: encoder–decoder attention plus masking, positions, FFNs and post-norm residual blocks. / 架构：编码器—解码器注意力，结合掩码、位置、FFN 与后归一化残差块。
- Evidence: strong translation results and informative ablations, with task and protocol limits. / 证据：较强翻译结果与有用消融，同时存在任务及协议边界。

### English speaker notes

Return to the opening question. Attention can replace sequence-aligned recurrence as the main contextual mechanism in this encoder–decoder model, and the reported experiments make a strong case for that architecture. But it does not eliminate all sequential work, all other network components, or the need for task-specific evidence. Ask the audience to explain one attention row without using the word “importance”: it is a compatibility-based, normalized weighting of values. Then ask them to distinguish the training mask from generation order. Close by naming a supported result and a limitation, rather than ending on a claim that the architecture solves every task. The remaining ten minutes are reserved for questions.

### 中文讲稿

回到开场：在该编码器—解码器中，注意力可以替代沿序列递推的循环，成为主要上下文机制，实验有力支持这一架构。但它没有消除所有顺序计算、其他网络结构或具体任务证据的必要性。请听众不用“重要性”一词解释一行注意力：它基于兼容度，将值向量按归一化权重混合。再区分训练掩码与生成顺序。结尾说出一个被支持的结果和一个局限，避免“解决所有任务”的概括。其后保留十分钟答疑。

### Audience prompt / 课堂互动

One-minute recap: define Q/K/V, explain masking, and give one experiment with its split and metric.

一分钟复述：定义 Q/K/V，解释掩码，并给出一个含划分与指标的实验。

## 10-minute Q&A bank

### 1. Did this paper invent attention? / 本文发明了注意力吗？

No. Its contribution is the Transformer transduction architecture; the paper itself reviews earlier attention and self-attention systems.

没有。贡献是 Transformer 序列转换架构；作者明确回顾了已有注意力和自注意力系统。

Source: §2, p.2

### 2. Why divide by the square root of key dimension? / 为何除以键维度的平方根？

Under the stated independence and unit-variance assumptions, dot-product variance is d_k. Scaling controls its standard deviation and reduces extreme softmax saturation.

在独立且单位方差假设下，点积方差为 d_k；缩放控制标准差，缓解 Softmax 过度饱和。

Source: §3.2.1, p.4

### 3. Are attention weights probabilities of the next word? / 注意力权重是下一个词的概率吗？

No. They normalize over key positions. The separate decoder output softmax normalizes over vocabulary items.

不是。注意力在键位置上归一化，解码器输出 Softmax 才在词表上归一化。

Source: §§3.2,3.4, pp.4–5

### 4. Are Q, K and V identical in self-attention? / 自注意力中的 Q、K、V 相同吗？

They share an input sequence, but normally use distinct learned projections. Their values need not be identical.

它们共用输入序列，但通常经过不同可学习投影，数值不必相同。

Source: §3.2.2, p.5

### 5. Why use multiple heads rather than one wide head? / 为何不用一个宽头，而用多头？

Different projections permit different mixtures in distinct subspaces, followed by concatenation and an output projection. Table 3 supports multiple heads in this setting, not an unlimited benefit from adding heads.

不同投影允许各子空间形成不同混合，再拼接与投影。表 3 支持此设置中的多头，但不支持无限增加头数。

Source: Table 3(A), p.9

### 6. What changes between self- and cross-attention? / 自注意力与交叉注意力有什么区别？

The source of Q versus K,V. Decoder cross-attention uses decoder queries and encoder-memory keys and values. The same matrix operator applies.

Q 与 K、V 的来源不同。解码器交叉注意力使用解码器查询和编码器记忆中的键、值，矩阵算子相同。

Source: §3.2.3, p.5

### 7. Why are target inputs shifted right? / 目标输入为何右移？

Together with the causal mask, shifting prevents a position from reading the very target it must predict. The allowed diagonal contains the preceding target token.

右移与因果掩码共同避免当前位置读取待预测答案，允许的对角线对应前一个目标词。

Source: §3.1, p.3

### 8. How can training be parallel but inference sequential? / 为何训练可并行、预测却顺序进行？

Teacher-forced target prefixes are known during training. During generation, each next token depends on an earlier token that must first be produced.

训练时教师强制的目标前缀已知；生成时下一个词依赖尚需产生的前一个词。

Source: §§3,3.1, pp.2–3

### 9. Why are positional encodings necessary? / 为何需要位置编码？

Unmasked self-attention without position signals is permutation-equivariant. Added position vectors expose order; sinusoidal encoding is one choice, not a uniqueness result.

没有位置信号的无掩码自注意力具有置换等变性；加入位置向量可提供顺序，正弦位置并非唯一选择。

Source: §3.5, p.6; derivation in guide

### 10. Is the original Transformer pre-norm? / 原始 Transformer 是前归一化吗？

No. The paper uses LayerNorm(x + Sublayer(x)), with dropout on the sublayer output during training. Modern variants may differ.

不是。本文使用 LayerNorm(x + Sublayer(x))，训练时对子层输出做 Dropout；现代变体可能不同。

Source: §§3.1,5.4, pp.3,8

### 11. Does O(1) path length imply constant runtime? / O(1) 路径长度代表恒定运行时间吗？

No. Dense attention still computes n² interactions, plus projection and feed-forward work. Path length, work and wall-clock latency are distinct.

不代表。稠密注意力仍计算 n² 对交互，还包含投影与前馈计算。路径长度、工作量和实际时延不同。

Source: Table 1, pp.6–7

### 12. Why is EN–FR reported as both 41.0 and 41.8? / 英法为何同时出现 41.0 和 41.8？

The selected v7 text is internally inconsistent: §6.1 prose says 41.0, while the abstract and Table 2 say 41.8. We use the table value and disclose the discrepancy; the source does not resolve it.

选定 v7 内部不一致：§6.1 正文为 41.0，摘要和表 2 为 41.8。讲义用表值并披露差异，原文没有解释这一差异。

Source: pp.1,8

### 13. Is 25.8 versus 28.4 a clean base/big comparison? / 25.8 与 28.4 是干净的基础／大模型对比吗？

No. 25.8 is a Table 3 dev score without checkpoint averaging; 28.4 is a Table 2 test score. Use matching splits and evaluation settings.

不是。25.8 是表 3 不做检查点平均的开发集值；28.4 是表 2 测试值。应使用同划分同协议。

Source: Tables 2–3, pp.8–9

### 14. Do attention maps explain model decisions? / 注意力图能解释模型决策吗？

They visualize a computation, but causal claims require additional evidence. The toy maps here are computed from declared synthetic matrices, not learned linguistic explanations.

它们显示某项计算，因果论断仍需额外证据。本讲义图来自已声明的合成矩阵，并非学得的语言学解释。

Source: Appendix, pp.13–15; critical interpretation

### 15. Does parsing prove zero-shot transfer? / 句法分析证明零样本迁移吗？

No. Separate parsing models were trained. This supports architectural reuse, not frozen-weight zero-shot transfer from translation.

不证明。作者训练了单独的句法分析模型，支持架构复用，不等于冻结翻译权重后的零样本迁移。

Source: §6.3, pp.9–10

### 16. How does this fit COMP6710 Area 2? / 它如何对应 COMP6710 Area 2？

It is foundational machinery for later transfer and few-shot systems. Direct evidence belongs to papers such as BERT or GPT-3; the 2017 paper itself is not a zero/few-shot experiment, and course acceptance is not established here.

它是后续迁移和少样本系统的基础机制。直接证据属于 BERT、GPT-3 等后续论文；2017 本文不是零／少样本实验，本讲义也不代表课程选题已获批准。

Source: §7; linked BERT/GPT-3 sources

### 17. What follow-up experiment would you propose? / 你会提出什么后续实验？

Match data, tokenization, parameter count, training budget and decoding. Run multiple seeds; measure BLEU, runtime and memory across lengths, then ablate heads and positional information separately. This is a proposed study, not a reported result.

匹配数据、分词、参数量、训练预算和解码；跑多个种子，随长度测 BLEU、时延和显存，再分别消融头与位置信息。这是建议研究，不是已报告结果。

Source: Authored critique of §§4–6

### 18. Can changing V affect attention weights? / 改变 V 会改变注意力权重吗？

With Q,K and the mask fixed, weights stay unchanged, but the weighted output changes. In an entire trained model, shared upstream parameters can couple these quantities.

固定 Q、K 和掩码时，权重不变，加权输出会改变；完整训练模型中的共享上游参数可能让它们相互关联。

Source: Eq.1, p.4

## References

- [Original paper v7](https://arxiv.org/pdf/1706.03762v7)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT-3](https://arxiv.org/abs/2005.14165)
- [ViT](https://arxiv.org/abs/2010.11929)

AI assisted organization, translation, illustration and code. Check the source and rehearse; the planned timings include demonstration and discussion, not a guaranteed verbatim reading duration.