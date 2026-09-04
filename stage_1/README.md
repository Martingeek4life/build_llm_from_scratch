# Stage 1 — Data Preparation & Sampling Pipeline

> Chapter 2: Working with Text Data
> *"Build a Large Language Model (From Scratch)"* — Sebastian Raschka

---

## Overview

Stage 1 focuses on the **data preparation pipeline** — everything that happens *before* the model sees any data.
The core idea is: raw text cannot be fed directly into an LLM. It must be converted into numbers first.

The pipeline goes through the following steps:

```
Raw text  →  Tokenization  →  Token IDs  →  Token Embeddings  →  LLM
```

---

## Section 2.2 — Tokenizing Text

![Figure 2.4 — Text processing pipeline in the context of an LLM](../assets/stage1.jpeg)

> *Figure 2.4 — A view of the text processing steps in the context of an LLM.
> Here, we split an input text into individual tokens, which are either words or
> special characters, such as punctuation characters.*

### What the figure shows

Starting from the bottom and going up:

| Step | Description |
|------|-------------|
| **Input text** | Raw string — e.g. `"This is an example."` |
| **Tokenized text** | The string is split into individual tokens: `["This", "is", "an", "example", "."]` |
| **Token IDs** | Each token is mapped to an integer via a vocabulary: `[40134, 2052, 133, 389, 12]` |
| **Token embeddings** | Each token ID is converted into a dense vector (a row of floating-point numbers) |
| **GPT-like decoder-only transformer** | The model processes the sequence of embeddings |
| **Postprocessing steps** | The model output is converted back into readable text |
| **Output text** | The final generated or processed text |

---

## Key Concepts

- **Tokenization** — Splitting text into units (tokens) the model can work with. Tokens can be words, subwords, or individual characters depending on the tokenizer.
- **Vocabulary** — A fixed mapping from tokens to integer IDs. GPT-2 uses a vocabulary of ~50,000 tokens.
- **Token embeddings** — Dense vector representations of token IDs. These are *learned* during training, not fixed.
- **Decoder-only Transformer** — The architecture used by GPT-style models. It processes tokens left-to-right and predicts the next token.

---

## Why this matters

Without a proper tokenization and embedding pipeline, the model has no way to process text.
This stage is the **foundation** of the entire LLM — get it wrong and nothing downstream works correctly.
