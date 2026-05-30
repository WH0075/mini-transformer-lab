# MiniTransformer-Lab

MiniTransformer-Lab is a small educational project that implements a character-level decoder-only Transformer language model from scratch using PyTorch.

MiniTransformer-Lab 是一个小型教学项目，使用 PyTorch 从零实现一个字符级 decoder-only Transformer 语言模型。

The goal of this project is to understand the core components of a Transformer by implementing them step by step, instead of directly using high-level libraries such as HuggingFace Transformers.

本项目的目标不是直接调用 HuggingFace Transformers 等高级库，而是通过逐步手写核心模块，理解 Transformer 语言模型的基本结构和训练流程。

---

## 1. Project Overview / 项目概述

This project builds a tiny character-level language model.

本项目实现了一个小型字符级语言模型。

The full training pipeline is:

完整训练流程如下：

```text
raw text
-> CharTokenizer
-> token ids
-> CharLanguageModelDataset
-> TinyTransformerLanguageModel
-> logits
-> CrossEntropyLoss
-> optimizer update
-> generated text
```

中文理解：

```text
原始文本
-> 字符级分词器
-> token id 序列
-> 语言模型数据集
-> TinyTransformerLanguageModel
-> logits 预测分数
-> 交叉熵损失
-> 优化器更新参数
-> 生成文本
```

The model learns to predict the next character from a short text corpus.

模型的训练目标是：根据前面的字符预测下一个字符。

For example:

例如：

```text
input:  h e l l
target: e l l o
```

---

## 2. Features / 项目功能

* Character-level tokenizer
  字符级 tokenizer

* Character-level language modeling dataset
  字符级语言模型数据集

* Scaled dot-product attention
  缩放点积注意力

* Causal mask for decoder-only language modeling
  用于 decoder-only 语言模型的因果掩码

* Multi-head self-attention
  多头自注意力机制

* Feed-forward network
  前馈神经网络

* Pre-LN Transformer block
  Pre-LN 结构的 TransformerBlock

* Tiny decoder-only Transformer language model
  小型 decoder-only Transformer 语言模型

* Training script for next-character prediction
  下一个字符预测训练脚本

* Checkpoint saving
  checkpoint 训练存档保存

* Unit tests with pytest
  使用 pytest 编写单元测试

---

## 3. Project Structure / 项目结构

```text
mini-transformer-lab/
├── data/
│   └── raw/
│       └── tiny_corpus.txt
├── src/
│   ├── __init__.py
│   └── minitransformer/
│       ├── __init__.py
│       ├── tensor_basics.py
│       ├── linear_regression_demo.py
│       ├── mlp.py
│       ├── train_mlp.py
│       ├── tokenizer.py
│       ├── datasets.py
│       ├── attention.py
│       ├── blocks.py
│       ├── model.py
│       └── train_lm.py
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_attention.py
│   ├── test_blocks.py
│   └── test_model.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Core Modules / 核心模块说明

### 4.1 `tokenizer.py`

Implements a simple character-level tokenizer.

实现一个简单的字符级 tokenizer。

It builds:

它会构建：

* `stoi`: character to integer id
  字符到整数 id 的映射

* `itos`: integer id to character
  整数 id 到字符的映射

* `vocab_size`: number of unique characters
  词表大小，也就是不同字符的数量

It supports:

支持：

```python
encode(text)
decode(ids)
```

Example:

示例：

```text
text: "hello"
ids:  [7, 4, 10, 10, 13]
```

---

### 4.2 `datasets.py`

Implements `CharLanguageModelDataset`.

实现 `CharLanguageModelDataset`。

For a token sequence, it creates input-target pairs.

对于一段 token 序列，它会构造语言模型训练所需的输入和目标。

```text
x = current sequence
y = next-character target sequence
```

中文理解：

```text
x = 当前输入序列
y = 右移一位后的目标序列
```

Example:

示例：

```text
x: h e l l
y: e l l o
```

The goal is next-character prediction.

训练目标是预测下一个字符。

---

### 4.3 `attention.py`

Implements the attention components.

实现 attention 相关核心组件。

Main components:

主要组件：

* `create_causal_mask`
* `scaled_dot_product_attention`
* `MultiHeadSelfAttention`

Causal mask prevents the model from seeing future tokens.

Causal mask 用来防止模型看到未来 token，避免训练时偷看答案。

For multi-head self-attention:

多头自注意力的核心形状：

```text
input:             [B, T, C]
output:            [B, T, C]
attention weights: [B, H, T, T]
```

Where:

其中：

```text
B = batch size
T = sequence length
C = embedding dimension
H = number of attention heads
```

Multi-head attention splits the hidden dimension into multiple heads, not the text sequence.

多头注意力切分的是 hidden dimension，不是把文本序列切成几段。

---

### 4.4 `blocks.py`

Implements:

实现：

* `FeedForward`
* `TransformerBlock`

The FeedForward network structure is:

FeedForward 的结构是：

```text
Linear(embed_dim, feedforward_dim)
-> GELU
-> Linear(feedforward_dim, embed_dim)
```

The Transformer block uses a Pre-LN structure:

TransformerBlock 使用 Pre-LN 结构：

```text
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

中文理解：

```text
先 LayerNorm，再进入 Attention，然后做残差连接；
再 LayerNorm，进入 FeedForward，然后做残差连接。
```

The block keeps the input and output shapes the same.

TransformerBlock 会保持输入输出形状一致：

```text
input:  [B, T, C]
output: [B, T, C]
```

This makes it possible to stack multiple Transformer blocks.

这样多个 TransformerBlock 才能连续堆叠。

---

### 4.5 `model.py`

Implements `TinyTransformerLanguageModel`.

实现 `TinyTransformerLanguageModel`。

The model structure is:

模型结构如下：

```text
input_ids
-> token embedding
-> position embedding
-> TransformerBlock x num_layers
-> final LayerNorm
-> lm_head
-> logits
```

Input and output shapes:

输入输出形状：

```text
input_ids: [B, T]
logits:    [B, T, vocab_size]
```

The last dimension of logits is `vocab_size`, because each position predicts a score for every token in the vocabulary.

`logits` 的最后一维是 `vocab_size`，因为每个位置都要对词表中的所有 token 给出预测分数。

---

### 4.6 `train_lm.py`

Trains the Tiny Transformer language model on `data/raw/tiny_corpus.txt`.

用于在 `data/raw/tiny_corpus.txt` 上训练 Tiny Transformer 语言模型。

The script:

该脚本会完成：

* reads text
  读取文本

* builds tokenizer
  构建 tokenizer

* builds dataset and dataloader
  构建 dataset 和 dataloader

* creates the model
  创建模型

* trains with CrossEntropyLoss
  使用 CrossEntropyLoss 训练

* saves a checkpoint
  保存 checkpoint

* generates sample text
  生成简单文本样本

---

## 5. Installation / 安装方式

Create and activate a virtual environment:

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

安装依赖：

```bash
pip install -r requirements.txt
```

If your system uses `python` instead of `python3`, you can also run:

如果你的系统中 `python` 已经指向 Python 3，也可以使用：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Run Tests / 运行测试

Run all tests:

运行全部测试：

```bash
python -m pytest
```

Run specific test files:

运行指定测试文件：

```bash
python -m pytest tests/test_tokenizer.py -v
python -m pytest tests/test_dataset.py -v
python -m pytest tests/test_attention.py -v
python -m pytest tests/test_blocks.py -v
python -m pytest tests/test_model.py -v
```

---

## 7. Train the Language Model / 训练语言模型

Make sure the corpus exists:

确认语料文件存在：

```bash
cat data/raw/tiny_corpus.txt
```

Run training:

运行训练脚本：

```bash
python -m src.minitransformer.train_lm
```

The training script prints:

训练脚本会输出：

* device information
  当前使用 CPU 还是 GPU

* corpus length
  语料长度

* vocabulary size
  词表大小

* dataset size
  数据集大小

* training loss
  训练损失

* generated text samples
  生成文本样本

The checkpoint is saved to:

checkpoint 默认保存到：

```text
checkpoints/tiny_transformer_lm.pt
```

The checkpoint is ignored by Git by default.

checkpoint 默认不提交到 Git。

---

## 8. Key Tensor Shapes / 核心张量形状

```text
input_ids: [B, T]
targets:   [B, T]

token_emb: [B, T, C]
pos_emb:   [T, C]
x:         [B, T, C]

attention weights:        [B, H, T, T]
TransformerBlock output:  [B, T, C]

logits: [B, T, vocab_size]

CrossEntropyLoss input:
logits reshaped to  [B*T, vocab_size]
targets reshaped to [B*T]
```

Where:

其中：

```text
B = batch size
T = sequence length
C = embedding dimension
H = number of attention heads
```

---

## 9. Training Result Example / 训练结果示例

With a very small corpus, the model can quickly overfit and generate text that resembles the training data.

在很小的语料上，模型会很快过拟合，并生成接近训练语料风格的文本。

Example training loss:

训练损失示例：

```text
Epoch [001/050], train_loss: 3.233872
Epoch [010/050], train_loss: 1.845415
Epoch [020/050], train_loss: 0.914133
Epoch [030/050], train_loss: 0.411575
Epoch [040/050], train_loss: 0.260079
Epoch [050/050], train_loss: 0.220762
```

Example generated text:

生成文本示例：

```text
hello pytorch
hello transformer
mini transformer
language models predic
```

This does not mean the model truly understands English. It mainly shows that the training pipeline works and that the model has learned local character patterns from a tiny corpus.

这并不代表模型真正理解了英语，而是说明训练链路已经打通，模型已经从小语料中学到了一些局部字符模式。

---

## 10. Checkpoint / 训练存档

A checkpoint is a training snapshot.

checkpoint 可以理解为训练存档。

It saves:

它会保存：

* model parameters
  模型参数

* optimizer state
  优化器状态

* training configuration
  训练配置

* tokenizer mappings
  tokenizer 的字符和 id 映射

* epoch and train loss
  当前训练轮数和训练损失

The checkpoint file is saved as:

checkpoint 文件保存为：

```text
checkpoints/tiny_transformer_lm.pt
```

This file is useful for future loading, generation, or continued training.

它可以用于后续加载模型、生成文本或继续训练。

---

## 11. Learning Notes / 学习收获

Through this project, I learned:

通过这个项目，我学习了：

* how token ids are mapped to embeddings;
  token id 如何映射为 embedding 向量；

* how position embeddings are added to token embeddings;
  position embedding 如何加入 token embedding；

* how causal mask prevents future information leakage;
  causal mask 如何防止未来信息泄露；

* how scaled dot-product attention works;
  scaled dot-product attention 的计算过程；

* how multi-head attention splits the hidden dimension into multiple heads;
  multi-head attention 如何把 hidden dimension 拆成多个 head；

* how TransformerBlock combines attention, feed-forward network, LayerNorm, and residual connections;
  TransformerBlock 如何组合 attention、FeedForward、LayerNorm 和 residual connection；

* how logits are trained with CrossEntropyLoss for next-token prediction;
  logits 如何通过 CrossEntropyLoss 进行下一个 token 预测训练；

* how to build a minimal language model training pipeline in PyTorch.
  如何用 PyTorch 搭建一个最小语言模型训练管线。

---

## 12. Limitations / 项目局限

This is a tiny educational implementation.

这是一个小型教学实现。

It is trained on a very small character-level corpus and is not intended to be a production language model.

它是在非常小的字符级语料上训练的，不是生产级语言模型。

The purpose is to understand the core mechanics of Transformer language models through a minimal PyTorch implementation.

本项目的目的，是通过最小 PyTorch 实现理解 Transformer 语言模型的核心机制。

---

## 13. Future Work / 后续可扩展方向

Possible future improvements include:

后续可以扩展：

* Add dropout
  加入 dropout

* Add validation split
  加入验证集划分

* Add model loading and inference script
  增加模型加载与推理脚本

* Add temperature and top-k sampling
  增加 temperature 和 top-k 采样

* Train on a larger corpus
  使用更大的语料训练

* Add README figures or architecture diagrams
  在 README 中加入结构图

* Refactor training configuration into YAML
  将训练配置整理为 YAML 文件

---

## 14. Summary / 总结

MiniTransformer-Lab implements a complete tiny Transformer language modeling pipeline from scratch.

MiniTransformer-Lab 从零实现了一条完整的小型 Transformer 语言模型训练链路。

The project covers:

本项目覆盖：

```text
text
-> tokenizer
-> dataset
-> attention
-> transformer block
-> language model
-> training loop
-> checkpoint
-> generation
```

It is designed for learning, debugging, and understanding how decoder-only Transformer language models work internally.

它适合用于学习、调试和理解 decoder-only Transformer 语言模型的内部工作机制。
