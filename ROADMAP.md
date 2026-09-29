# 学习路线

目标：看懂大模型从预训练到后训练的每一步，并能用评测判断模型在金融、法律任务上的表现。

## 阶段 0：打基础（进行中）

- [x] [00 注意力机制的由来：seq2seq 与 attention](notes/00-seq2seq-attention.md)
- [x] [01 The Illustrated Transformer 核心笔记](notes/01-attention.md)
- [ ] 02 反向传播与损失函数（Karpathy：micrograd、makemore 第 1 部分）
- [ ] 03 从零搭一个 GPT（Karpathy：Let's build GPT）
- [ ] 04 [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [ ] 05 [DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)
- [ ] 建立各家训练方法对比表

视频资料：[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)（Andrej Karpathy）

## 阶段 1：数据与评测

评测设计、题目泄露、用模型当评委、标注一致性、数据构建与质量控制。

资料：[AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)（Chip Huyen）第 3–4 章、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

## 阶段 2：动手训练

PyTorch 基础、transformers 与 TRL、LoRA 微调、偏好对齐（DPO）。

资料：[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)、[TRL 文档](https://huggingface.co/docs/trl/index)、[RLHF Book](https://rlhfbook.com/)（Nathan Lambert）

## 阶段 3：深化

预训练数据、对齐、用强化学习训练推理能力。

资料：[Stanford CS336](https://cs336.stanford.edu/)、各家最新技术报告

## 待查概念

从经典 Transformer 到现在的大模型，改动了哪些地方：

- [ ] 只用解码器的结构
- [ ] 旋转位置编码 RoPE
- [ ] 混合专家架构 MoE
- [ ] 后训练：监督微调、偏好对齐、强化学习
