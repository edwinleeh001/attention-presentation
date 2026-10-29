# Transformer 与大模型 32 问 / 32 questions on Transformers and language models

整理日期 / Prepared: 2026-10-29

题目按用户提供的三张截图逐题识别；Q16、Q25 由重叠截图补齐。答案为独立编写的教学解释，保留截图编号；不是原论文自带问答。
Questions transcribed from three user-supplied screenshots; overlaps complete Q16 and Q25. Answers are independently written teaching explanations, not a Q&A from the paper.

范围：原论文核心与后续 LLM 扩展分别标记。保留原有 20 章、18 问答与 30 分钟展示节奏，本题库供复习和扩展答疑。
Scope: original-paper content is distinguished from later LLM extensions. This revision bank supplements the existing 20 chapters and 18 paper questions; it is outside the timed talk.

## 模型结构 / Architecture

### Q1. Transformer主要由哪两大核心模块构成？

What are the two main modules of a Transformer?

范围 / Scope: 原论文核心 / Original paper

**答案**：原论文的整体结构是编码器（Encoder）和解码器（Decoder）：前者把源序列转成上下文表示，后者结合源表示与已生成的目标前缀预测下一个词元。若问的是“每层的两类核心子层”，答案是多头注意力与逐位置前馈网络；原始解码层还包含交叉注意力。

**Answer:** The original model has an encoder that represents the source sequence and a decoder that predicts target tokens from those representations and the target prefix. At the sublayer level, the key building blocks are multi-head attention and a position-wise feed-forward network; each original decoder layer also has cross-attention.

**易错点**：注意题目在问整体结构还是层内组件。现代 Transformer 也可以仅用编码器或仅用解码器。

**Watch out:** Distinguish the whole architecture from its sublayers. Encoder-only and decoder-only variants also exist.

**术语 / Terms:** Encoder（编码器） · Decoder（解码器） · Sublayer（子层）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q2. 自注意力机制里的 Q、K、V 分别是什么？

What do Q, K and V mean in self-attention?

范围 / Scope: 原论文核心 / Original paper

**答案**：Q（Query，查询）表示当前位置要寻找什么；K（Key，键）用于与查询计算匹配程度；V（Value，值）提供被汇聚的信息。自注意力中三者由同一序列表示 X 经不同的可学习线性投影得到。每一行查询对所有允许的键计算分数，经 softmax 得到权重，再对值加权求和。

**Answer:** Queries describe what each position seeks; keys determine matching scores; values carry the information to aggregate. In self-attention, all three come from the same sequence representation X through different learned projections. Each query scores allowed keys, softmax normalizes the scores, and the resulting weights combine the values.

公式 / Formula: `Q=XWQ, K=XWK, V=XWV; Attention(Q,K,V)=softmax(QKᵀ/√dₖ + M)V`

**易错点**：Q、K、V 不是固定的三个词，也不是三个独立输入句子；交叉注意力的 Q 与 K/V 则来自不同序列。

**Watch out:** Q, K and V are learned representations, not three fixed words. Cross-attention uses different sequences for Q and K/V.

**术语 / Terms:** Query（查询） · Key（键） · Value（值）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q3. 多头注意力的核心优势在哪？

What is the main advantage of multi-head attention?

范围 / Scope: 原论文核心 / Original paper

**答案**：多个头使用不同的投影，在不同表示子空间中并行计算注意力，再拼接并投影回模型维度。这样同一词元可以同时汇聚不同位置和关系的信息，减少单次加权平均的表达限制。固定总维度时，每个头的维度通常相应减小。

**Answer:** Different heads attend in different learned representation subspaces. Their outputs are concatenated and projected back to the model width, allowing a token to combine several relationships instead of relying on one weighted average. With fixed total width, each head is typically narrower.

**易错点**：不能保证每个头都有明确的语法分工；更多头也不保证更好，需要消融验证。

**Watch out:** Heads need not have neatly interpretable roles, and adding heads does not guarantee improvement.

**术语 / Terms:** Attention head（注意力头） · Representation subspace（表示子空间）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q4. 位置编码解决什么问题？常见实现方式有哪些？

Why do we need positional information, and what forms can it take?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：不带位置线索的自注意力对输入排列是等变的，无法仅凭内容区分“谁先谁后”。原论文将正弦/余弦位置向量加到词嵌入上，也比较了可学习的绝对位置嵌入。后续方法包括相对位置表示和旋转位置编码（RoPE）；后两者是后续发展，不是 2017 年原论文的方法。

**Answer:** Without positional cues, self-attention is permutation equivariant: content alone does not identify order. The paper adds sinusoidal position vectors to token embeddings and also compares learned absolute embeddings. Later alternatives include relative position representations and rotary position embeddings (RoPE).

**易错点**：位置编码不自动保证超出训练长度仍表现良好；RoPE 也不是简单把位置向量加到输入上。

**Watch out:** Positional information does not guarantee length extrapolation. RoPE is not an additive input embedding.

**术语 / Terms:** Positional encoding（位置编码） · Absolute position（绝对位置） · RoPE（旋转位置编码）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7); [Shaw et al. · Self-Attention with Relative Position Representations (2018)](https://arxiv.org/abs/1803.02155); [Su et al. · RoFormer / Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)

### Q5. Encoder和Decoder结构上最本质的区别是什么？

How do the original encoder and decoder differ structurally?

范围 / Scope: 原论文核心 / Original paper

**答案**：原始编码器在有效源词元之间使用双向自注意力。原始解码器先做因果自注意力，只读取目标前缀，再通过编码器—解码器交叉注意力读取源序列表示，最后做前馈变换。因此它既有访问范围的限制，又多了一类连接源信息的注意力子层。

**Answer:** The original encoder uses bidirectional self-attention over valid source tokens. The decoder uses causal self-attention over the target prefix, then encoder–decoder cross-attention to the source representations, followed by a feed-forward network. It differs both in visibility and in its extra cross-attention sublayer.

**易错点**：不能把“所有 Decoder 都有交叉注意力”当作通则；常见仅解码器语言模型没有该子层。

**Watch out:** Cross-attention is part of the original decoder, but not required in every decoder-only language model.

**术语 / Terms:** Bidirectional attention（双向注意力） · Causal attention（因果注意力） · Cross-attention（交叉注意力）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q6. 为什么要加残差连接和层归一化？去掉会怎样？

Why use residual connections and layer normalization? What if they are removed?

范围 / Scope: 原论文核心 / Original paper

**答案**：残差连接提供恒等通路，让子层学习对已有表示的修正，并改善深层梯度传播。层归一化在每个词元的特征维上调整均值和方差，使表示尺度更可控。原论文采用后归一化。去掉其中一项通常会使深层优化更难、收敛更不稳定或性能下降，具体影响需实验判断。

**Answer:** Residual connections preserve an identity path and let a sublayer learn a correction, improving gradient propagation. Layer normalization normalizes features within each token representation to control activation scale. The original model uses post-normalization. Removing either often makes deep optimization harder or less stable, but the outcome depends on the setup.

公式 / Formula: `LayerNorm(x + Sublayer(x))`

**易错点**：LayerNorm 不等于跨样本的 BatchNorm；去掉它们并不意味着任何模型都绝对无法训练。

**Watch out:** LayerNorm is not BatchNorm. Removing these mechanisms does not make every possible model untrainable.

**术语 / Terms:** Residual connection（残差连接） · Layer normalization（层归一化） · Post-LN（后归一化）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q7. 处理长序列时，Transformer比RNN强在哪？

Where does a Transformer improve on an RNN for long sequences?

范围 / Scope: 原论文核心 / Original paper

**答案**：自注意力让任意两个可见位置在一层内直接交互，缩短远距离信息传播路径；训练时，已知输入和目标前缀可以并行计算各位置。RNN 则沿时间步递推。但标准稠密注意力的分数矩阵随长度 n 按 n² 增长，长上下文可能昂贵；自回归生成也仍需逐步进行。

**Answer:** Self-attention provides direct interactions between visible positions, shortening long-range paths. Training can process positions in parallel when inputs and target prefixes are known, while an RNN has sequential recurrence. However, dense attention has quadratic score-matrix size in sequence length, and autoregressive decoding remains sequential.

公式 / Formula: `Dense attention scores: n × n per head; pairwise attention work: O(n²d)`

**易错点**：“更容易建模长距离关系”不等于“任意长序列都更省显存或更快”。

**Watch out:** Shorter dependency paths do not imply lower memory use or faster execution at every sequence length.

**术语 / Terms:** Long-range dependency（长距离依赖） · Parallel computation（并行计算） · Quadratic complexity（二次复杂度）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

## 关键组件 / Components

### Q8. 前馈网络在Transformer里承担什么角色？

What does the feed-forward network do in a Transformer?

范围 / Scope: 原论文核心 / Original paper

**答案**：注意力主要在词元之间交换信息，FFN 则对每个位置的特征进行非线性变换与组合。原论文使用两层线性变换夹 ReLU，先从 512 维扩到 2048 维，再压回 512 维；同一层的各位置共享这组参数，不同层通常不共享。

**Answer:** Attention exchanges information across tokens; the FFN nonlinearly transforms features within each token. The original base model uses two linear maps with ReLU, expanding from 512 to 2048 dimensions and projecting back to 512. Positions share FFN parameters within a layer, while different layers normally have different parameters.

公式 / Formula: `FFN(x)=max(0,xW₁+b₁)W₂+b₂`

**易错点**：“逐位置”不代表每个位置训练一套独立权重；单独的 FFN 不混合不同词元。

**Watch out:** Position-wise does not mean separate weights for every position. The FFN itself does not mix tokens.

**术语 / Terms:** Feed-forward network（前馈网络） · Position-wise transformation（逐位置变换）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q9. 注意力掩码主要有哪两种？各自作用是什么？

What are the two common attention masks, and what do they do?

范围 / Scope: 原论文核心 / Original paper

**答案**：填充掩码（padding mask）阻止有效词元关注补齐长度用的 PAD 键；因果掩码（causal mask）阻止位置 i 关注未来位置 j>i。概念上，在 softmax 前把禁止位置的分数加上 −∞，使其权重为 0。两者可以组合。PAD 查询的输出和损失通常还要单独忽略。

**Answer:** A padding mask prevents attention to artificial PAD keys. A causal mask prevents position i from attending to future positions j>i. Conceptually, blocked scores receive negative infinity before softmax, giving them zero probability. Both masks can be combined; padded query outputs and loss positions need separate handling.

公式 / Formula: `Mᵢⱼ=0 when allowed, −∞ when blocked`

**易错点**：不同 API 对布尔 True 的含义可能相反；实现时应核对文档，避免整行都被屏蔽而产生 NaN。

**Watch out:** Boolean mask conventions differ across APIs. Avoid unintentionally masking every key in a row.

**术语 / Terms:** Padding mask（填充掩码） · Causal mask（因果掩码）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q10. 词嵌入维度一般怎么定？有经验参考吗？

How should embedding dimensionality be chosen?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：维度是模型容量与计算、显存预算之间的设计选择，没有由词表大小直接决定的万能公式。原论文 base 的 d_model=512，big 为 1024；常见等宽多头实现要求 d_model 可被头数 h 整除。训练新模型时结合数据量、任务和验证集选择；使用预训练模型时通常沿用其既定维度。

**Answer:** Embedding width trades capacity against compute and memory; vocabulary size alone does not determine it. The original base and big models use widths of 512 and 1024. Standard equal-width head implementations require divisibility by the number of heads. For a new model, compare widths under a fixed budget; for a pretrained model, normally keep its existing width.

公式 / Formula: `Embedding parameters ≈ vocabulary size × embedding width`

**易错点**：这些数值是论文配置，不是适合所有任务的推荐值；词表扩大也会增加嵌入参数量。

**Watch out:** These are historical configurations, not universal recommendations. A larger vocabulary also increases embedding parameters.

**术语 / Terms:** Embedding dimension（嵌入维度） · Model width（模型宽度）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q11. Dropout为什么要用？设太大或太小会怎样？

Why use dropout, and what happens when its rate is too high or too low?

范围 / Scope: 原论文核心 / Original paper

**答案**：Dropout 在训练时随机置零部分激活，降低对特定特征组合的依赖，作为正则化帮助泛化。丢弃率过大会破坏学习信号、导致欠拟合或训练变慢；过小则可能不足以抑制过拟合。原论文 base 使用 0.1，但应以验证集效果选择。推理时关闭 dropout。

**Answer:** Dropout randomly zeros activations during training to reduce reliance on particular feature combinations. Too much can impair the learning signal or cause underfitting; too little may provide insufficient regularization. The original base model uses 0.1, but validation should guide the choice. Dropout is disabled for ordinary inference.

**易错点**：丢弃率 p 和保留率 1−p 不要混淆；更多训练数据或不同模型可能需要不同强度。

**Watch out:** Do not confuse drop probability p with keep probability 1−p. The right rate depends on the data and model.

**术语 / Terms:** Dropout（随机失活） · Regularization（正则化） · Underfitting（欠拟合）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q12. Decoder训练阶段的“掩码”具体怎么工作？为什么不能看后面的词？

How does masking work during decoder training, and why hide future tokens?

范围 / Scope: 原论文核心 / Original paper

**答案**：先将目标序列右移：输入 [BOS, y₁, y₂]，对应预测 [y₁, y₂, y₃]。位置 i 可以看自己及之前的输入，但看不到未来输入，因此不能直接抄到要预测的答案。训练虽能并行计算所有位置，仍满足与推理一致的因果条件；若只掩码而不正确右移，可能泄漏标签。

**Answer:** Shift targets right: inputs [BOS, y₁, y₂] predict [y₁, y₂, y₃]. At each position, causal attention allows the current and earlier inputs but blocks future ones. This prevents copying the answer and matches the information available at inference, while still enabling parallel training. Masking alone does not replace correct target shifting.

**易错点**：因果掩码通常允许主对角线，因为该位置输入已右移；它不同于 BERT 随机遮盖词元。

**Watch out:** The diagonal is normally allowed because inputs are shifted. This differs from BERT-style masked-token prediction.

**术语 / Terms:** Right-shifted targets（目标右移） · Teacher forcing（教师强制） · Label leakage（标签泄漏）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q13. 预训练模型的核心设计思想是什么？跟传统训练最大区别在哪？

What is the idea behind pretraining, and how does it differ from training from scratch?

范围 / Scope: 后续扩展 / Later extension

**答案**：先在较大、较广的数据上学习可迁移的表示或生成能力，再用较少的任务数据微调，或通过提示词调用已有能力。与每个任务都从随机参数开始训练相比，主要差异是复用已学参数、数据和计算。预训练常用自监督目标；具体策略也可包含监督信号。

**Answer:** Pretraining learns reusable representations or generation skills from broad data. A downstream task can then adapt those parameters with less task-specific data or use prompting. Compared with training each task from scratch, the key difference is reuse of learned weights, data and compute. Self-supervision is common, but not the only possible pretraining signal.

**易错点**：提示词中的上下文学习通常不更新权重；2017 年 Transformer 论文主要研究监督机器翻译，并非现代通用 LLM 预训练。

**Watch out:** In-context learning usually changes no weights. The 2017 paper studied supervised translation, not modern general-purpose LLM pretraining.

**术语 / Terms:** Pretraining（预训练） · Transfer learning（迁移学习） · Fine-tuning（微调）

**依据 / Sources:** [Devlin et al. · BERT (2018/2019)](https://arxiv.org/abs/1810.04805); [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

## 核心概念 / Core concepts

### Q14. 模型参数数量和计算量有什么本质区别？哪个更能衡量规模？

How do parameter count and computation differ? Which measures scale better?

范围 / Scope: 后续扩展 / Later extension

**答案**：参数量是存储的可学习数值个数，反映模型容量和权重存储需求；计算量是执行训练或推理所需的运算，取决于参数、词元数、序列长度、架构和执行方式。衡量模型大小可报参数量，衡量训练投入则应报 FLOPs 与数据量，部署还要看延迟、吞吐和显存；不存在单一通用尺度。

**Answer:** Parameter count measures stored learned values, indicating model capacity and weight storage. Computation measures operations performed and depends on tokens, sequence length, architecture and execution as well as parameters. Report parameters for model size and FLOPs plus data volume for training investment; deployment also needs latency, throughput and memory.

**易错点**：同样参数量不代表同样成本；MoE 的总参数量与每词元激活参数量也不同。

**Watch out:** Equal parameter counts do not imply equal cost. In a mixture of experts, total and active parameters differ.

**术语 / Terms:** Parameter count（参数量） · FLOPs（浮点运算量） · Throughput（吞吐量）

**依据 / Sources:** [Kaplan et al. · Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361); [Hoffmann et al. · Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)

### Q15. 生成式 AI 是什么？典型任务有哪些？

What is generative AI, and what are typical tasks?

范围 / Scope: 后续扩展 / Later extension

**答案**：生成式 AI 学习数据分布或条件分布，依据提示、上下文或随机变量产生新样本。典型任务包括文本续写、翻译与摘要、代码生成、图像合成、语音与音乐生成。输入可以是文字，也可以是图像、音频等；不同任务可采用 Transformer、扩散模型等不同架构。

**Answer:** Generative AI learns a data distribution or conditional distribution and produces new samples from prompts, context or random inputs. Tasks include text continuation, translation, summarization, code generation, image synthesis and audio generation. Both conditioning inputs and outputs can be multimodal, and the model need not be a Transformer.

**易错点**：“生成新内容”不代表内容必然真实、原创或正确；仍要按任务验证。

**Watch out:** Producing new content does not guarantee truth, originality or correctness; evaluation must match the task.

**术语 / Terms:** Generative AI（生成式人工智能） · Conditional distribution（条件分布）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7); [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

### Q16. 大模型为什么需要海量训练数据？数据不足会怎样？

Why do large models need substantial training data? What if data is insufficient?

范围 / Scope: 后续扩展 / Later extension

**答案**：大量且多样的数据帮助估计语言规律、覆盖长尾情况，并让高容量模型获得足够训练信号。数据不足、重复过多或覆盖面狭窄时，模型可能记忆样本、过拟合，或能力未被充分训练。应同时考虑参数规模、训练词元量、数据质量和计算预算，而不是只追求更大文件。

**Answer:** Large, diverse datasets provide statistical evidence, cover rare cases and supply enough training signal for high-capacity models. Too little, repetitive or narrow data can encourage memorization, overfitting or undertraining. Parameters, training tokens, data quality and compute should be considered jointly, rather than maximizing raw file size alone.

**易错点**：更多低质量或重复数据并不必然更好；固定预算下，模型和数据需要共同配置。

**Watch out:** More low-quality or duplicate data is not automatically better. Allocate model size and data together under a fixed budget.

**术语 / Terms:** Training tokens（训练词元） · Data coverage（数据覆盖） · Compute-optimal（计算最优）

**依据 / Sources:** [Hoffmann et al. · Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)

### Q17. 什么是过拟合？实际中怎么判断？

What is overfitting, and how can we detect it?

范围 / Scope: 后续扩展 / Later extension

**答案**：过拟合指模型过度适应训练样本，包括偶然噪声，导致对未见数据泛化较差。常见迹象是训练损失持续下降，而独立验证损失上升或任务指标变差。应使用无泄漏、与目标场景相符的划分，多次检查学习曲线；验证集用于选择模型，最终测试集保留到评估时使用。

**Answer:** Overfitting occurs when a model adapts too closely to training examples, including noise, and generalizes poorly. A typical sign is decreasing training loss alongside rising held-out validation loss or worsening task metrics. Use leakage-free, representative splits and learning curves; tune on validation data and reserve the test set for final evaluation.

**易错点**：验证表现差也可能来自分布偏移、标注错误或评估实现问题，不能只凭一个数断言过拟合。

**Watch out:** Poor validation results can also reflect distribution shift, label errors or evaluation bugs.

**术语 / Terms:** Overfitting（过拟合） · Generalization（泛化） · Validation set（验证集）

**依据 / Sources:** [Kaplan et al. · Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361)

### Q18. “模型推理”具体指哪个过程？和训练有什么区别？

What is model inference, and how does it differ from training?

范围 / Scope: 后续扩展 / Later extension

**答案**：推理是用已经学到的参数处理输入、计算输出，例如分类或逐词元生成。训练通常包含前向计算、损失、反向传播和优化器更新；普通推理主要执行前向计算及解码，不更新模型权重。推理时 KV 缓存会随上下文变化，但这是运行状态，不是训练出的参数被修改。

**Answer:** Inference uses learned parameters to process inputs and produce predictions or generated tokens. Training normally includes a forward pass, loss computation, backpropagation and optimizer updates; ordinary inference uses forward computation and decoding without updating weights. A growing KV cache is runtime state, not a weight update.

**易错点**：这里的 inference 是执行模型，不保证模型具有可靠的逻辑推理能力。

**Watch out:** Running inference does not by itself imply reliable logical reasoning.

**术语 / Terms:** Inference（推理执行） · Backpropagation（反向传播） · KV cache（键值缓存）

**依据 / Sources:** [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

### Q19. 怎么理解大模型的“涌现能力”？能举个例子吗？

What are “emergent abilities” in large models? Can you give an example?

范围 / Scope: 后续扩展 / Later extension

**答案**：该说法描述某项任务在较小模型上几乎不成功，到较大规模时突然表现明显改善的观测，例如多位数算术的整题正确率。但“突然”可能受评估指标影响：要求答案完全匹配时，许多逐步改善的局部能力会被压成接近 0，跨过阈值后才显现。应同时查看连续指标和不同模型规模。

**Answer:** The term describes observed tasks that seem absent in smaller models but improve sharply at larger scale, such as exact-match multi-digit arithmetic. Apparent abruptness can depend on the metric: gradual improvements in partial correctness may remain invisible until an exact-answer threshold is crossed. Examine continuous metrics and multiple scales before inferring a discontinuity.

**易错点**：这是存在争议的经验现象，不能据此证明模型出现意识，或保证扩大规模就获得某项能力。

**Watch out:** This is a contested empirical interpretation, not evidence of consciousness or a guarantee of new skills.

**术语 / Terms:** Emergent ability（涌现能力） · Exact match（精确匹配） · Metric threshold（度量阈值）

**依据 / Sources:** [Wei et al. · Emergent Abilities of Large Language Models (2022)](https://arxiv.org/abs/2206.07682); [Schaeffer et al. · Are Emergent Abilities of Large Language Models a Mirage? (2023)](https://arxiv.org/abs/2304.15004)

### Q20. 解释一下“缩放定律”，对大模型发展有什么意义？

What are scaling laws, and why do they matter for large models?

范围 / Scope: 后续扩展 / Later extension

**答案**：缩放定律是某些训练条件下观察到的经验关系：验证损失随模型参数、数据量或计算量增加，常呈近似幂律改善，并出现边际收益递减。它帮助预测资源投入的回报、分配固定预算。后续 Chinchilla 研究强调参数量和训练词元量应共同增长；具体最优比例依赖数据、目标和成本假设。

**Answer:** Scaling laws are empirical regularities: under particular conditions, validation loss improves approximately as a power law in parameters, data or compute, often with diminishing returns. They help estimate returns and allocate budgets. Chinchilla emphasizes jointly scaling parameters and training tokens; optimal ratios depend on data, objectives and cost assumptions.

**易错点**：它们不是保证所有任务都按同一曲线进步的自然定律，也不能替代数据质量与实际评估。

**Watch out:** They do not guarantee the same improvement curve for every task, nor replace data-quality checks and evaluation.

**术语 / Terms:** Scaling laws（缩放定律） · Power law（幂律） · Diminishing returns（边际收益递减）

**依据 / Sources:** [Kaplan et al. · Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361); [Hoffmann et al. · Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)

## 训练与评估 / Training & evaluation

### Q21. 预训练的核心目标是什么？常用损失函数有哪些？

What is the goal of pretraining, and which loss functions are common?

范围 / Scope: 后续扩展 / Later extension

**答案**：目标是从大规模数据中学到可复用的统计规律与表示。自回归语言模型通常最小化下一词元的负对数似然，即分类交叉熵；掩码语言模型在选定遮盖位置计算词元预测交叉熵；去噪文本模型学习重建目标文本。不同模态或方法也可使用对比损失、噪声预测均方误差等，应按训练目标选择。

**Answer:** Pretraining learns reusable statistical structure and representations. Causal language models minimize next-token negative log-likelihood, equivalent to token classification cross-entropy. Masked language models compute token prediction loss at selected positions; text denoising reconstructs target text. Other modalities and objectives may use contrastive losses or noise-prediction mean squared error.

公式 / Formula: `L = −(1/T) Σₜ log pθ(xₜ | x<t)`

**易错点**：库的交叉熵函数常接收原始 logits，内部已处理 log-softmax；不要机械地重复 softmax。

**Watch out:** Many cross-entropy APIs expect raw logits and apply log-softmax internally; do not blindly apply softmax twice.

**术语 / Terms:** Negative log-likelihood（负对数似然） · Cross-entropy（交叉熵） · Self-supervised learning（自监督学习）

**依据 / Sources:** [Devlin et al. · BERT (2018/2019)](https://arxiv.org/abs/1810.04805); [Hugging Face · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)

### Q22. 语言模型的自回归生成是怎么回事？优缺点？

How does autoregressive generation work? What are its strengths and weaknesses?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：它把序列概率分解成“给定已有前缀时下一个词元概率”的乘积。生成时选择一个词元，追加到前缀，再预测下一步，直到 EOS 或长度上限。优点是概率建模清晰、适合可变长生成；缺点是步骤依赖造成延迟，早期错误可能影响后续。已知真实前缀的训练可以并行，并不矛盾。

**Answer:** Autoregression factors sequence probability into next-token probabilities conditioned on the prefix. Generation selects a token, appends it, and repeats until EOS or a length limit. It provides a natural probabilistic model for variable-length sequences, but sequential dependence adds latency and early mistakes can influence later tokens. Training can still parallelize with known target prefixes.

公式 / Formula: `p(x₁…xT)=∏ₜ p(xₜ | x<t)`

**易错点**：自回归描述条件依赖，不等同于贪心选择；同一模型可以用采样、贪心或束搜索解码。

**Watch out:** Autoregression describes conditional dependence, not a particular token-selection rule.

**术语 / Terms:** Autoregression（自回归） · End-of-sequence token（序列结束符）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7); [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

### Q23. 数据并行训练的基本原理是什么？多卡怎么同步梯度？

How does data-parallel training work, and how are gradients synchronized?

范围 / Scope: 后续扩展 / Later extension

**答案**：各卡持有模型副本，处理不同数据分片，分别前向和反向，然后用 all-reduce 聚合梯度，通常取平均；各副本以相同优化器状态和聚合梯度更新，保持参数一致。需正确分片数据与同步更新步。梯度累积可以减少同步次数并扩大有效批量；不同样本数或有效词元数时，要核对加权与损失归一化。

**Answer:** Each device holds a model replica and computes gradients on a different data shard. All-reduce aggregates gradients, usually by averaging; replicas then apply matching optimizer updates. Data sharding and update steps must align. Gradient accumulation can reduce synchronization frequency and increase effective batch size; unequal sample or token counts require correct weighting and loss normalization.

公式 / Formula: `Effective batch = per-device microbatch × devices × accumulation steps (equal-size case)`

**易错点**：DDP 主要分摊数据工作量，普通 DDP 不会自动把单个模型的参数拆开来节省每卡权重显存。

**Watch out:** Ordinary DDP splits data work; it does not automatically shard model parameters across devices.

**术语 / Terms:** Data parallelism（数据并行） · All-reduce（全规约） · Gradient accumulation（梯度累积）

**依据 / Sources:** [PyTorch · Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Q24. Batch Size太大或太小对训练有什么影响？怎么调？

How does batch size affect training, and how should it be tuned?

范围 / Scope: 后续扩展 / Later extension

**答案**：小批量的梯度噪声通常更大、设备利用率可能较低；大批量更占显存，在固定训练词元预算下意味着更少的更新步，通常需要重新调学习率与 warmup，超过某个规模后收益可能减小。先找显存允许且吞吐合适的微批量，再用梯度累积控制全局批量，结合验证曲线和实际速度调整。

**Answer:** Smaller batches usually produce noisier gradients and may underutilize hardware. Larger batches use more memory and yield fewer updates for a fixed token budget, often requiring learning-rate and warmup changes. Choose a workable microbatch, use accumulation to set global batch size, and evaluate both validation behavior and throughput.

**易错点**：“批量越大越好”或“越小泛化越好”都不是通则；语言模型最好同时报告每步有效词元数。

**Watch out:** Neither larger nor smaller batches universally generalize better. For language models, also report effective tokens per update.

**术语 / Terms:** Microbatch（微批量） · Global batch（全局批量） · Warmup（预热）

**依据 / Sources:** [PyTorch · Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html); [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q25. 学习率设置不当会出现什么问题？太大或太小分别有什么现象？

What happens when the learning rate is too high or too low?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：学习率太大可能使损失震荡、发散或出现非有限数；太小则更新缓慢，在预算内难以充分学习。观察训练和验证曲线、梯度范数及数值异常，结合优化器、批量和调度选择。原论文采用 Adam，并先预热学习率再随步数下降，base 配置 warmup 为 4000 步。

**Answer:** An excessive learning rate may cause oscillation, divergence or non-finite values. A very small rate may learn too slowly within the available budget. Inspect loss curves, gradient norms and numerical issues while tuning the optimizer, batch and schedule together. The original paper uses Adam with warmup followed by decay, including 4,000 warmup steps in its base setup.

**易错点**：损失平台期不一定只是学习率问题，也可能是数据、容量或实现错误；4000 步不是通用常数。

**Watch out:** A plateau can also reflect data, capacity or implementation issues. Four thousand warmup steps is not a universal rule.

**术语 / Terms:** Learning rate（学习率） · Learning-rate schedule（学习率调度） · Gradient norm（梯度范数）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

### Q26. 训练损失持续下降，就代表模型真的在变好吗？

Does steadily falling training loss mean the model is really improving?

范围 / Scope: 后续扩展 / Later extension

**答案**：它只直接说明模型对训练目标、训练数据的拟合在改善。验证表现可能因过拟合变差，数据泄漏可能让评估虚高，而预测损失也不等同于事实准确性、任务成功率或人类偏好。应同时检查独立验证损失、任务指标、典型失败案例及部署场景表现。

**Answer:** It directly indicates better fitting of the training objective on training data. Validation can worsen through overfitting, leakage can inflate evaluation, and prediction loss does not directly measure factual accuracy, task success or human preference. Track held-out loss, task metrics, failure cases and performance in the intended setting together.

**易错点**：比较训练曲线前应确认数据、损失归一化、批量与标签平滑设置是否一致。

**Watch out:** Before comparing curves, check data, loss normalization, batch settings and label smoothing.

**术语 / Terms:** Training loss（训练损失） · Task metric（任务指标） · Data leakage（数据泄漏）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7); [Hugging Face · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)

### Q27. 评估生成质量的常用指标有哪些？各自侧重什么？

Which metrics evaluate generation quality, and what does each emphasize?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：文本生成中，BLEU 偏重与参考答案的 n-gram 精确匹配并惩罚过短输出；ROUGE 常用于摘要中的词面覆盖；BERTScore 用上下文向量衡量语义相似。困惑度衡量真实文本的预测概率。实际任务还需事实核验、代码测试、任务成功率及人工评估；图像和音频则需要相应模态的指标。

**Answer:** For text, BLEU emphasizes reference n-gram precision with a brevity penalty; ROUGE commonly measures lexical coverage in summaries; BERTScore measures contextual semantic similarity. Perplexity evaluates predictive probability on reference text. Real tasks also need factual checks, execution tests, task success and human assessment. Image and audio generation require modality-specific metrics.

**易错点**：任何单一分数都不能概括质量；词面或语义相似不保证事实正确，自动评审也有偏差。

**Watch out:** No single score captures quality. Lexical or semantic similarity does not establish factual correctness.

**术语 / Terms:** BLEU（机器翻译指标） · ROUGE（摘要覆盖指标） · BERTScore（语义相似指标）

**依据 / Sources:** [Papineni et al. · BLEU (ACL 2002)](https://aclanthology.org/P02-1040/); [Zhang et al. · BERTScore (2019/2020)](https://arxiv.org/abs/1904.09675); [Hugging Face · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity); [Lin · ROUGE (2004)](https://aclanthology.org/W04-1013/)

### Q28. 困惑度反映的是模型哪方面能力？和生成质量一定正相关吗？

What does perplexity measure? Does it necessarily track generation quality?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：困惑度是每词元平均负对数似然的指数，反映模型对给定测试文本的概率预测能力；在相同评估条件下通常越低越好。但它不直接评估生成文本的事实性、帮助程度或解码后的表现。不同分词器、数据或上下文窗口下的数值不能直接横比；标准困惑度主要用于自回归语言模型。

**Answer:** Perplexity is the exponential of average token negative log-likelihood. It measures probability assigned to held-out text; lower is generally better under matched conditions. It does not directly assess generated factuality, usefulness or decoding outcomes. Values across different tokenizers, datasets or context windows are not directly comparable; standard perplexity is primarily for autoregressive models.

公式 / Formula: `PPL = exp[−(1/T) Σₜ log pθ(xₜ | x<t)] (natural logarithm)`

**易错点**：与生成质量没有必然的单调关系。原论文中标签平滑会使困惑度变差，却改善准确率和 BLEU。

**Watch out:** There is no necessary monotonic relation to generation quality. The paper reports worse perplexity but better accuracy and BLEU with label smoothing.

**术语 / Terms:** Perplexity（困惑度） · Tokenizer（分词器） · Label smoothing（标签平滑）

**依据 / Sources:** [Hugging Face · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity); [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7)

## 应用与优化 / Applications & optimization

### Q29. 贪心解码和集束搜索的核心区别是什么？适用场景？

How do greedy decoding and beam search differ, and when are they useful?

范围 / Scope: 原论文 + 实践延伸 / Paper + practice

**答案**：贪心解码每一步只保留当前概率最大的词元，速度快但容易错过整体得分更好的序列。集束搜索（束搜索）维护 B 个候选前缀，按累计对数概率及可能的长度惩罚筛选，再继续扩展。翻译等较受约束的输出可尝试束搜索；开放创作常用采样来获得多样性，最终应以任务评估决定。

**Answer:** Greedy decoding keeps the highest-probability next token at each step. Beam search maintains B candidate prefixes and ranks expansions by cumulative log-probability, often with length penalties. It may help constrained generation such as translation, while sampling is often useful for open-ended diversity. Select decoding settings using task-specific evaluation.

**易错点**：束搜索是近似搜索，不保证找到全局最优，更不保证人类认为最好的答案；更大 B 不一定更好。

**Watch out:** Beam search is approximate and does not guarantee the best sequence or the best answer for people. Larger beams are not always better.

**术语 / Terms:** Greedy decoding（贪心解码） · Beam search（集束搜索） · Length penalty（长度惩罚）

**依据 / Sources:** [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7); [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

### Q30. 温度参数怎么影响生成结果？设为0或很大分别会怎样？

How does temperature affect generation? What about zero or very high values?

范围 / Scope: 后续扩展 / Later extension

**答案**：采样时将 logits 除以正温度 T 再做 softmax：T<1 使分布更尖锐，T>1 更平坦。T 趋近 0 时概率集中到最大 logit；T 很大时，在其他限制不变且 logits 有限的候选集合上趋向均匀。API 的 T=0 通常是切换到贪心的特殊约定，不能直接代入除法公式。

**Answer:** For sampling, divide logits by positive temperature T before softmax. T<1 sharpens the distribution; T>1 flattens it. As T approaches zero, mass concentrates on maximal logits; very large T approaches uniformity over finite allowed logits before other filters. An API setting of zero is commonly a special greedy mode, not literal division by zero.

公式 / Formula: `pᵢ(T)=exp(zᵢ/T) / Σⱼ exp(zⱼ/T), T>0`

**易错点**：固定候选集下，正温度不改变 argmax 的排序，因此纯贪心通常不受影响；低温也不保证事实正确。

**Watch out:** Positive temperature preserves argmax ordering for fixed candidates, so ordinary greedy selection is unchanged. Low temperature does not guarantee truth.

**术语 / Terms:** Temperature（温度） · Sampling（采样） · Logits（原始分数）

**依据 / Sources:** [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies); [Hugging Face · Generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation)

### Q31. 推理时为什么要限制最大生成长度？不限制会有什么后果？

Why cap generation length, and what can happen without a limit?

范围 / Scope: 后续扩展 / Later extension

**答案**：长度上限提供确定的资源与时间边界。模型不一定及时输出 EOS，可能重复或长时间生成，增加延迟、费用和 KV 缓存占用，也可能触及上下文窗口限制。应区分“最多新生成多少词元”和“输入加输出总长度”，同时设置 EOS/停止条件，并按任务给足回答空间。

**Answer:** A length limit bounds time and resource use. A model may not emit EOS promptly and can repeat or continue generating, increasing latency, cost and KV-cache memory or reaching the context limit. Distinguish maximum new tokens from total input-plus-output length, combine limits with stop conditions, and leave enough room for the task.

**易错点**：上限是兜底，不是质量保证；设得过短会截断答案，完成状态应明确显示是否因长度停止。

**Watch out:** A cap is a safeguard, not a quality guarantee. Short caps can truncate answers; distinguish length stops from normal completion.

**术语 / Terms:** Maximum new tokens（最大新词元数） · Context window（上下文窗口） · Stopping criteria（停止条件）

**依据 / Sources:** [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies); [Hugging Face · Generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation)

### Q32. 大模型微调的主流方法有哪些？LoRA、Adapter各有什么特点？

What are common fine-tuning methods, and how do LoRA and adapters compare?

范围 / Scope: 后续扩展 / Later extension

**答案**：常见方式有全参数微调、只更新部分层/任务头，以及参数高效微调（PEFT），如 LoRA、Adapter、提示/前缀微调。LoRA 冻结原权重，以低秩矩阵 BA 学习增量，适用时可合并回权重，减少额外推理开销。Adapter 在网络中加入小型可训练瓶颈模块，便于切换任务，但通常增加前向运算。二者都主要减少可训练参数和优化器状态。

**Answer:** Options include full fine-tuning, updating selected layers or task heads, and parameter-efficient methods such as LoRA, adapters and prompt/prefix tuning. LoRA freezes base weights and learns a low-rank update BA, which can often be merged for inference. Adapters insert small trainable bottleneck modules, enabling task switching but typically adding forward operations. Both reduce trainable parameters and optimizer state.

公式 / Formula: `W′=W+BA; B∈ℝ^(d_out×r), A∈ℝ^(r×d_in); trainable parameters=r(d_in+d_out)`

**易错点**：PEFT 不等于训练零成本：仍需运行基座并保存必要激活。LoRA 的秩、目标层和数据会影响效果，不能保证与全量微调相同。

**Watch out:** PEFT still runs the backbone and stores needed activations. LoRA rank, target layers and data affect quality; parity with full fine-tuning is not guaranteed.

**术语 / Terms:** PEFT（参数高效微调） · LoRA（低秩适配） · Adapter（适配器）

**依据 / Sources:** [Hu et al. · LoRA: Low-Rank Adaptation of Large Language Models (2021/2022)](https://arxiv.org/abs/2106.09685); [Houlsby et al. · Parameter-Efficient Transfer Learning for NLP (2019)](https://arxiv.org/abs/1902.00751)

## 来源核验范围 / Source reading scope

- [Vaswani et al. · Attention Is All You Need (2017), arXiv v7](https://arxiv.org/html/1706.03762v7) — Sections 3–6; original architecture, training and experiments
- [Devlin et al. · BERT (2018/2019)](https://arxiv.org/abs/1810.04805) — Abstract; pretraining and transfer
- [Kaplan et al. · Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361) — Abstract; empirical scaling with parameters, data and compute
- [Hoffmann et al. · Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556) — Abstract; joint allocation of parameters and training tokens
- [Wei et al. · Emergent Abilities of Large Language Models (2022)](https://arxiv.org/abs/2206.07682) — Definition and examples of apparent emergence
- [Schaeffer et al. · Are Emergent Abilities of Large Language Models a Mirage? (2023)](https://arxiv.org/abs/2304.15004) — Abstract; discontinuous metrics and alternative explanations
- [PyTorch · Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) — DDP gradient synchronization and process model; accessed 2026-10-29
- [Hugging Face · Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity) — Definition, tokenization and context-window limitations; accessed 2026-10-29
- [Hugging Face · Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) — Greedy, beam search and sampling; accessed 2026-10-29
- [Hu et al. · LoRA: Low-Rank Adaptation of Large Language Models (2021/2022)](https://arxiv.org/abs/2106.09685) — Abstract; frozen backbone and low-rank updates
- [Houlsby et al. · Parameter-Efficient Transfer Learning for NLP (2019)](https://arxiv.org/abs/1902.00751) — Abstract; trainable adapters with a shared frozen backbone
- [Papineni et al. · BLEU (ACL 2002)](https://aclanthology.org/P02-1040/) — Original metric reference
- [Zhang et al. · BERTScore (2019/2020)](https://arxiv.org/abs/1904.09675) — Abstract; contextual token similarity
- [Su et al. · RoFormer / Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864) — Abstract; rotational position representation
- [Shaw et al. · Self-Attention with Relative Position Representations (2018)](https://arxiv.org/abs/1803.02155) — Abstract; relative positions in attention
- [Lin · ROUGE (2004)](https://aclanthology.org/W04-1013/) — Original metric reference
- [Hugging Face · Generation configuration](https://huggingface.co/docs/transformers/en/main_classes/text_generation) — Temperature and maximum new token configuration; accessed 2026-10-29

说明：这是概念讲解与原始出处核对，未训练模型或复现论文基准；经验性的调参建议需在具体任务上验证。
These are conceptual explanations and source checks, not model training or benchmark reproduction. Tuning advice requires validation in the intended setting.
