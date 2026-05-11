# Future Mask Experiment Instruction (Refined)

## Overview

This experiment investigates whether the **future mask**, when combined with the standard causal mask across different attention heads in the encoder, encodes positional information more effectively than the causal mask alone — and whether this translates into improved language modeling performance. The experiment extends the NoPos framework (Haviv et al., 2022) and is conducted on a fork of the NoPos repository: `https://github.com/Anxinal/futureMask.git`.

The encoder–decoder architecture is chosen because the encoder naturally processes the full input sequence bidirectionally, making it a valid setting for future-mask experiments without the information leakage that arises in decoder-only models (where future-masked heads can see the prediction target during training).

All models use the **NoPos setting** (`--no-token-positional-embeddings`), meaning positional information must be inferred entirely from the attention mask structure.

---

## 1. Architecture

**Encoder–Decoder Transformer** with **8 attention heads** and **no positional encoding**.

- 8-layer multihead encoder–decoder transformer.
- The encoder self-attention layers have their masks modified per the comparison conditions in Section 4.
- The decoder self-attention retains standard causal masking (8C) in all conditions; the cross-attention is unmasked as usual.
- This separation is key: the encoder sees the full input and is where mask variations are tested, while the decoder's causal mask ensures autoregressive generation remains valid.

**Input–output split:** Since WikiText-103 is a language modeling corpus (not a natural seq2seq dataset), the input must be divided between encoder and decoder. Use a **prefix-based split**: for each sequence of length *T*, the encoder receives the first *k* tokens as the source, and the decoder autoregressively predicts tokens *k+1* through *T* as the target (conditioned on the encoder output via cross-attention and on previously generated tokens via causal self-attention).

- **Specify *k*:** e.g. *k = T/2* (fixed half-split), or *k* drawn uniformly from [T/4, 3T/4] per sample (randomised split for robustness). State the chosen strategy.
- At inference/evaluation time the decoder generates tokens one by one, attending only to the encoder output and its own causal history.

**Hyperparameters to specify** (fill in or match the NoPos defaults):

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Encoder layers         | 8                              |
| Decoder layers         | 8                              |
| Attention heads        | 8                              |
| Embedding dim          | 1024 |
| FFN dim                | 2048 |
| Dropout                | 0.3         |
| Sequence length (*T*)  | 512        |
| Encoder prefix (*k*)   | T/2        |
| Optimizer              | Adam      |
| Learning rate          | default                   |
| Training updates       | default                  |
| Batch size (tokens)    | default                   |

---

## 2. Dataset and Tasks

**Dataset:** WikiText-103 (Merity et al., 2017).

### Task A: Next-Token Prediction (Language Modeling)

Conditional language modeling in the encoder–decoder setup. The encoder processes the prefix (first *k* tokens); the decoder autoregressively predicts the remaining tokens conditioned on the encoder output. At each decoder position *t*, the model predicts the next target token. Evaluation metric: **perplexity** (= exp(cross-entropy loss)) computed over the decoder target tokens.

### Task B: Position Prediction (Probing)

After training the language model, freeze the model weights and train a linear probe on the **encoder** hidden representations to predict the absolute position of each token. This follows the probing methodology of Haviv et al. (2022) but applied to the encoder side, since the encoder is where mask variations are introduced.

**Probe details to specify:**
- Which layer's hidden states are used as input to the probe (e.g. final layer, or evaluate across all layers).
- Probe architecture (e.g. single linear layer mapping hidden state → position index).
- Evaluation metric (e.g. mean absolute error between predicted and true position).

---

## 3. Research Objectives

**Objective 1 (Positional encoding):** Does the future mask, compared to a plain causal mask, help the model encode positional information more effectively? Measured via the position prediction probe (Task B).

**Objective 2 (Model performance):** Does the future mask in the encoder improve language modeling performance (measured by decoder perplexity) compared to (a) causal-mask-only encoder and (b) fully bidirectional (no-mask) encoder configurations? Measured via perplexity on WikiText-103 (Task A).

---

## 4. Comparison Conditions

Notation for mask types applied per attention head:
- **C** = Causal mask (attend to positions ≤ current)
- **F** = Future mask (attend to positions > current; with `future_mask_allow_self=True` to retain self-attention at the diagonal)
- **B** = Bidirectional / no mask (attend to all positions)

### Encoder Self-Attention Mask Configurations

Replace the self-attention mask in each **encoder** layer's 8 heads as follows:

| Condition   | Head allocation           | Description                      |
|-------------|---------------------------|----------------------------------|
| 8B          | 8 bidirectional           | Baseline: standard encoder (no mask) |
| 8C          | 8 causal                  | All-causal encoder               |
| 4B4C        | 4 bidirectional + 4 causal| Mixed bidirectional-causal        |
| 2F2C4B      | 2 future + 2 causal + 4 bidirectional | Mixed with future mask |
| 4F4C        | 4 future + 4 causal       | Balanced future-causal            |
| 3F3C2B      | 3 future + 3 causal + 2 bidirectional | Majority masked        |

**Held constant across all conditions:**
- Decoder self-attention: fully causal (8C).
- Cross-attention: unmasked.
- All mask assignments are applied uniformly across all encoder layers (i.e. every encoder layer uses the same head-to-mask mapping).

---

## 5. Metrics and Reporting

For each condition, report:

- **Training loss** (cross-entropy): The standard cross-entropy loss on the decoder's target tokens, averaged over training batches.
- **Validation loss** (cross-entropy): Computed on the WikiText-103 validation set after training.
- **Perplexity**: exp(validation cross-entropy loss).
- **Position probe accuracy** (Task B only): Mean absolute error of the linear position probe trained on the frozen **encoder** hidden states, evaluated on validation data.

**Loss definition:** The loss is the cross-entropy loss computed over the decoder's target tokens. At each decoder position *t*, the model's output distribution is compared against the ground-truth next target token. The loss is averaged across all decoder positions and all sequences in the batch. The encoder mask configuration only affects how the encoder builds its representations of the source prefix; the decoder loss computation itself is identical across all conditions.

**State any code changes** made to the NoPos repository to implement these experiments, including the specific files modified and the nature of the changes.

---

## 6. Implementation Notes

- **Mask construction:** For the encoder, construct a 3D mask tensor of shape `[num_heads × T_enc × T_enc]` per layer, assigning each head its designated mask type (causal, future, or no mask). This tensor is passed directly to the encoder's multi-head attention module. The decoder self-attention uses a standard causal mask of shape `[T_dec × T_dec]` (shared across all heads). Cross-attention uses no mask.
- **Future mask definition:** M_{ij} = 0 if j > i, −∞ if j ≤ i. With `future_mask_allow_self=True`, M_{ii} = 0 (the diagonal is unmasked to avoid empty attention rows for the last token).
- **Random seeds:** Run each condition with at least 2–3 different random seeds and report mean ± standard deviation if resources permit.





Based on the description in @experiment_instruction_refined.md , edit the codebase from nopos such that the experiment described in the instruction can be carried out. After that, just carry a brief trial experiment with 100 updates(1 epoch) again and generate a breif summary report of the experiment result
