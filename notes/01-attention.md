# 01 The Illustrated Transformer 核心笔记

来源：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（Jay Alammar）

## 一、整体架构

经典 Transformer 是 Encoder–Decoder 架构：

Tokens → Embedding + Positional Encoding → Encoder → Decoder → Linear + Softmax → Output

Encoder 负责形成输入序列的上下文表示（contextualized representations）；Decoder 根据 Encoder 的信息和已经生成的 token，逐步预测下一个 token。

## 二、Embedding + Positional Encoding

Token 先被转换成 embedding，每个 token 是一个 $d\_{\text{model}}$ 维的向量。Self-Attention 本身不知道词序，所以还要加上位置编码：
```math
X = \text{Embedding} + \text{PE}
```

经典 Transformer 用不同频率的 sin/cos 函数生成 PE。具体数值不用记，核心是让模型知道 token 的位置和相对位置关系。

## 三、Self-Attention：Q、K、V

输入分别乘以三个训练出来的矩阵：
```math
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
```

完整公式：
```math
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

直觉上：

- Q 和 K：决定关注谁、关注多少
- V：真正被提取和传递的信息

先用 $QK^T$ 计算 token 之间的匹配程度，再用 softmax 得到注意力权重 $\alpha\_{ij}$，最后加权求和，把其他 token 的信息融合进当前 token：
```math
z_i = \sum_j \alpha_{ij} v_j
```

## 四、第一层和后续层的区别

- 第一层之前：输入只是 Embedding + PE，一个 token 还没有通过 Self-Attention 获得其他 token 的信息
- 经过第一层后：每个 token 的表示开始融合其他 token 的信息
- 所以后续层的 Q、K、V 已经来自带上下文的表示，而不只是原始 token embedding。这是深层 Transformer 能逐步形成复杂上下文关系的基础

## 五、Multi-Head Attention

同时拥有多套 $W\_i^Q, W\_i^K, W\_i^V$，不同的头在不同的表示子空间里做注意力，因此可以学到不同类型的关系。

注意：并不是人为规定"头 1 看主语、头 2 看指代"，而是训练中可能自然形成不同的关注模式。

## 六、Self-Attention 与 FFN

一个 Encoder Layer 的核心可以这样理解：

- **Attention**：token 之间交换信息
- **FFN**：每个 token 独立加工交换后得到的信息

FFN 对每个位置分别执行同一个非线性变换。它具体如何加工，后面学训练机制时再深入。

## 七、Residual + LayerNorm

每个子层都不是直接把旧表示替换掉，而是 $x + f(x)$：
```math
X' = \text{LayerNorm}(X + \text{Attention}(X))
```
```math
X'' = \text{LayerNorm}(X' + \text{FFN}(X'))
```

- Add 加的是当前子层的输入，而不是永远加最初的 embedding
- Residual 既有利于保留原信息，也有利于深层网络的梯度传播

## 八、Decoder

Decoder 比 Encoder 多两个特点，顺序是：

Masked Self-Attention → Cross-Attention → FFN

**Masked Self-Attention**

训练时完整答案其实已经存在，比如 "I am a student"。为了并行训练，同时又不让前面的 token 偷看后面的答案，会把未来位置的注意力分数设为 $-\infty$，softmax 之后对应权重就变成 0。

- 训练时：未来 token 存在，但不准看
- 推理时：未来 token 本来就还不存在

**Cross-Attention**

- Q 来自 Decoder，K 和 V 来自 Encoder
- 即 Decoder 根据"我现在需要什么信息"形成 Q，再去 Encoder 的表示里找相关内容
- 经典 Transformer 中，所有 Decoder Layer 的 Cross-Attention 都读取最后一层 Encoder 的输出，但各层有自己的注意力参数

## 九、Linear + Softmax

Decoder 最终输出的是隐藏表示，还不是词的概率。

- Linear（$hW + b$）：把 $d\_{\text{model}}$ 维的隐藏表示映射成词表大小维度的 logits
- Softmax：把 logits 转成每个 token 的概率

Hidden State → Linear → Logits → Softmax → Probability

Linear + Softmax 可以在概念上看成一个输出模块，但不能合并成一个普通矩阵，因为 softmax 是非线性的。

## 十、Training

- One-hot 编码（如 [0, 1, 0, 0, …]）只表示"正确答案是哪一个 token"，本身不表达语义，和 embedding 不同
- 训练的基本框架：Prediction → Loss → Backpropagation → Gradient → 参数更新
- 这篇文章只讲到这个直觉，还没有真正解释"为什么参数更新能让正确 token 的概率越来越高"，需要继续学：Cross-Entropy → Gradient → Backpropagation → Optimizer

## 我的理解与疑问

**Q1. Q/K 的匹配只取决于词本身吗？**

不是。第一层主要来自 token embedding 加位置信息；第一层注意力之后，表示已经包含上下文，所以后续层的 Q、K 也包含上下文信息。

**Q2. 第一层之前，一个 token 有上下文信息吗？**

如果"上下文信息"特指通过 Self-Attention 从其他 token 获得的信息，那么没有。此时它只有 Embedding + Positional Encoding，第一层注意力之后才开始真正带上下文。

**Q3. 为什么注意力权重最后乘在 V 上？**

Q/K 决定从谁那里取多少信息，V 才是真正被取走的信息，所以 $z\_i = \sum\_j \alpha\_{ij} v\_j$。

**Q4. Attention 和 FFN 是什么关系？**

Attention：和其他 token 交流。FFN：自己加工交流后得到的信息。

**Q5. Residual 的 Add 加谁？**

永远加当前子层的输入，即 $x + f(x)$，而不是每一层都加最初的 X。

**Q6. 未来 token 都还没生成，为什么需要 Mask？**

因为训练时未来 token 已经存在于标准答案里。Mask 防止模型训练时偷看未来；推理时未来 token 本来就不存在。

**Q7. 所有 Decoder Layer 都使用最后一层 Encoder 的输出吗？**

经典 Transformer 中是的。但不同 Decoder Layer 有各自的 Cross-Attention 参数，因此可以用不同方式读取同一份 Encoder 表示。

**Q8. 为什么 Linear 之后还需要 Softmax？**

Linear 把隐藏表示变成每个 token 的分数，Softmax 把分数变成概率。Softmax 是非线性的，不能被吸收进一个固定的线性矩阵。

**Q9. 文章已经解释了模型怎么学会让正确词的概率最高吗？**

还没有完全解释。目前只有 Prediction → Loss → Backpropagation → 调整参数 这个框架，真正理解还需要学 cross-entropy、梯度和优化器。

**Q10. Beam Search 把路径概率相乘，会不会破坏 causal mask？**

不会。路径概率是：
```math
P(w_1,w_2,w_3) = P(w_1)\,P(w_2 \mid w_1)\,P(w_3 \mid w_1,w_2)
```

每一步仍然只依赖已经生成的 token。Beam Search 只是生成阶段挑选候选路径的方法，不改变注意力内部的 causal mask。

## 一条主线
```text
Embedding + Position
        ↓
Q、K、V → Attention → Multi-Head
        ↓
Residual + Norm → FFN → Residual + Norm
        ↓
Encoder Representations
        ↓
Masked Self-Attention → Cross-Attention → FFN
        ↓
Linear → Softmax → P(next token)
```

最后分成两条路：

- 训练：P → Loss → Backprop → 更新参数
- 生成：P → Greedy / Beam / Sampling → Token
