# Transformer from Scratch — PyTorch

A decoder-only Transformer language model implemented and trained **from scratch in PyTorch**, without using high-level Transformer or multi-head attention APIs.

The project demonstrates the mathematical and implementation details behind Transformer-based autoregressive language modeling, including scaled dot-product attention, causal masking, multi-head attention, positional encoding, residual connections, LayerNorm, and next-token prediction.

---

## Project Overview

The goal of this project is to understand and implement the core architecture of a Transformer language model from first principles.

Instead of using:

```python
nn.Transformer
```

or:

```python
nn.MultiheadAttention
```

the core Transformer components are implemented manually.

### Pipeline

```text
Raw Text
    ↓
Character-level Tokenization
    ↓
Token Embeddings
    +
Sinusoidal Positional Encoding
    ↓
Transformer Block × 4
    ↓
Final LayerNorm
    ↓
Linear Language Model Head
    ↓
Next-token Logits
    ↓
Cross-Entropy Loss
```

---

## Architecture

The model uses the following configuration:

| Hyperparameter         |            Value |
| ---------------------- | ---------------: |
| Dataset                | Tiny Shakespeare |
| Tokenization           |  Character-level |
| Vocabulary             |   ~65 characters |
| Context Length         |              128 |
| Embedding Dimension    |              256 |
| Attention Heads        |                4 |
| Head Dimension         |               64 |
| Feed-Forward Dimension |             1024 |
| Transformer Layers     |                4 |
| Dropout                |              0.1 |
| Optimizer              |            AdamW |
| Learning Rate          |         3 × 10⁻⁴ |

---

## Mathematical Formulation

### 1. Token Embedding

Each token is mapped to a learnable vector:

$$
E \in \mathbb{R}^{V \times d_{model}}
$$

The input representation is:

$$
X = E_{token} + PE
$$

---

### 2. Positional Encoding

Sinusoidal positional encoding is used to represent token positions:

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

---

### 3. Self-Attention

The input is projected into queries, keys and values:

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Scaled dot-product attention is:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}+M
\right)V
$$

where \(M\) is the causal attention mask.

---

### 4. Multi-Head Attention

The attention operation is performed independently across multiple heads:

$$
MultiHead(X)
=
Concat(head_1,\ldots,head_h)W_O
$$

For this model:

$$
d_{model}=256
$$

$$
h=4
$$

$$
d_k=\frac{256}{4}=64
$$

---

### 5. Feed-Forward Network

Each Transformer block contains a position-wise feed-forward network:

$$
FFN(X)
=
W_2\,GELU(W_1X+b_1)+b_2
$$

The hidden dimension expands from:

$$
256\rightarrow1024\rightarrow256
$$

---

### 6. Residual Connections and LayerNorm

The Transformer uses residual connections:

$$
X_1=X+Attention(LN(X))
$$

and:

$$
X_2=X_1+FFN(LN(X_1))
$$

---

### 7. Language Modeling Objective

The model predicts the next token:

$$
P(x_t|x_1,\ldots,x_{t-1})
$$

using cross-entropy loss:

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P(x_t|x_{<t})
$$

---

## Training

The model is trained using:

* AdamW optimizer
* Learning rate: \(3\times10^{-4}\)
* Gradient clipping
* Causal self-attention
* Cross-entropy loss
* Training/validation evaluation

Training and validation loss are monitored throughout training.

---

## Results

The primary evaluation metrics are:

### Cross-Entropy Loss

Measures the difference between the predicted probability distribution and the actual next token.

Lower is better.

### Perplexity

$$
PPL=e^{\mathcal{L}}
$$

Lower perplexity indicates better next-token prediction.

Add your actual final values here after training:

| Model       | Layers | Parameters | Validation Loss | Validation PPL |
| ----------- | -----: | ---------: | --------------: | -------------: |
| Transformer |      2 |   `<fill>` |        `<fill>` |       `<fill>` |
| Transformer |      4 |   `<fill>` |        `<fill>` |       `<fill>` |

---

## Training Curves

![Training Loss](results/training_loss.png)

![Perplexity](results/perplexity.png)

The decrease in training and validation loss demonstrates that the model learns meaningful statistical patterns from the text corpus.

---

## Attention Visualization

Attention maps were extracted from different Transformer layers and heads to inspect how tokens interact with previous positions.

![Attention Map](results/attention_map.png)

Because the model is autoregressive, the causal mask prevents each token from attending to future positions.

---

## Ablation Study

A small depth ablation experiment compares 2-layer and 4-layer Transformer configurations.

### Research Question

> How does increasing Transformer depth affect language-model performance?

The models are compared using:

* Validation loss
* Validation perplexity
* Parameter count

The experiment provides a simple analysis of the trade-off between model capacity and performance.

---

## Text Generation

After training, the model can generate text autoregressively.

Example prompt:

```text
ROMEO:
```

Example generated output:

```text
<insert your actual generated sample here>
```

Generation uses temperature-controlled sampling:

$$
P_i=
softmax
\left(
\frac{z_i}{T}
\right)
$$

where \(T\) controls the randomness of the generated sequence.

---

## Key Engineering Components

The project implements:

* Character-level tokenization
* Token embeddings
* Sinusoidal positional encoding
* Query, Key and Value projections
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Feed-forward networks
* Layer normalization
* Residual connections
* Dropout
* Autoregressive language modeling
* AdamW optimization
* Gradient clipping
* Validation evaluation
* Perplexity calculation
* Text generation
* Attention visualization
* Model checkpointing

---

## Project Structure

```text
transformer-from-scratch/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Transformer_From_Scratch.ipynb
│
├── src/
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer_block.py
│   ├── model.py
│   └── train.py
│
├── results/
│   ├── training_loss.png
│   ├── perplexity.png
│   ├── attention_map.png
│   └── ablation_results.png
│
└── checkpoints/
    └── transformer_shakespeare.pt
```

---

## How to Run

Clone the repository:

```bash
git clone <your-repository-url>
cd transformer-from-scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/Transformer_From_Scratch.ipynb
```

and run the notebook cells sequentially.

---

## What I Learned

This project provided practical understanding of:

1. How self-attention computes contextual relationships between tokens.
2. Why the \(1/\sqrt{d_k}\) scaling factor is required.
3. How causal masking enables autoregressive language modeling.
4. How multiple attention heads learn independent representations.
5. The role of positional information in Transformers.
6. How residual connections and LayerNorm stabilize deep architectures.
7. How Transformer depth affects model capacity and performance.
8. How training loss and perplexity can be used to evaluate language models.

---

## Future Improvements

Potential extensions include:

* Implementing BPE tokenization
* Larger training corpora
* Learning-rate warmup and cosine decay
* Mixed-precision training
* Larger Transformer architectures
* Comparison with an LSTM baseline
* More extensive hyperparameter sweeps
* Better quantitative attention analysis
* Scaling experiments

---

## Technologies

**Python · PyTorch · NumPy · Matplotlib · Deep Learning · NLP · Transformers**
