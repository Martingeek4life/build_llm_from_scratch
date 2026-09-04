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

## The Problem

Neural networks are mathematical functions — they only operate on numbers.
Raw text is a sequence of characters: `"I HAD always thought Jack Gisburn rather a cheap genius..."`.
There is no direct way to feed this string into a matrix multiplication or a gradient descent update.

The challenge is: **how do you turn language into numbers in a way that preserves meaning?**
A naive approach (e.g. mapping each character to its ASCII code) loses all structure and context.
We need something smarter.

---

## The Intuition

Think of a dictionary. Every word has a unique entry number.
If you agree on the same dictionary with someone else, you can communicate using only numbers — and reconstruct the original message perfectly.

That is exactly what a **tokenizer + vocabulary** does:
- The tokenizer splits text into atomic units called **tokens** (words, subwords, punctuation)
- The vocabulary assigns a unique integer **ID** to each token
- The result is a sequence of integers the model can process mathematically

But integers alone are still too rigid — the number `42` is not "close" to `43` in any meaningful linguistic sense.
So we go one step further: each integer is mapped to a **dense vector** (token embedding), where similar tokens end up with similar vectors. This is where meaning begins to emerge.

---

## The Solution

The data preparation pipeline solves the problem in three steps:

**1. Tokenization**
Split the raw text into tokens using a tokenizer (e.g. a simple whitespace/punctuation splitter, or a BPE tokenizer like GPT-2's).

```python
# Example: "This is an example." → ["This", "is", "an", "example", "."]
```

**2. Vocabulary mapping (Token IDs)**
Build a vocabulary from the training corpus and map each token to a unique integer ID.

```python
# Example: {"This": 40134, "is": 2052, "an": 133, "example": 389, ".": 12}
# → [40134, 2052, 133, 389, 12]
```

**3. Token Embeddings**
Pass each token ID through an embedding layer that converts it into a dense floating-point vector.
These vectors are *learned* during training — they are not fixed.

```python
# Each token ID → vector of size d_model (e.g. 768 dimensions for GPT-2 small)
```

---

## Application

In `tokenizer.py`, the first concrete step of this pipeline is implemented:

```python
# Download the training text
urllib.request.urlretrieve(url, file_path)

# Read and inspect the raw data
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))  # → 20,479 characters
print(raw_text[:99])
# → "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no"
```

**What this tells us:**
- The dataset (*The Verdict* by Edith Wharton) contains **20,479 characters**
- It is a short story — small enough to experiment with on a laptop, large enough to demonstrate the full pipeline
- The next steps will tokenize this text, build a vocabulary, and convert it into token IDs ready for the model

---

## SimpleTokenizerV1 — Encode & Decode

![Encode and Decode flow](../assets/tokenizer_enc_dec.png)

> *Illustration of `tokenizer.encode(text)` (top) and `tokenizer.decode(ids)` (bottom)*

The `SimpleTokenizerV1` class implements the two core operations of any tokenizer:

### `encode(text)` — Text → Token IDs

```python
def encode(self, text):
    preprocessed = re.split(pattern, text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    IDs = [self.str_to_int[item] for item in preprocessed]
    return IDs
```

1. Split the input string into tokens using the regex pattern (whitespace + punctuation)
2. Strip and filter empty tokens
3. Map each token to its integer ID using `str_to_int` (the vocabulary)

→ `"The brown dog"` becomes `[7, 0, 1, ...]`

---

### `decode(ids)` — Token IDs → Text

```python
def decode(self, ids):
    text = " ".join([self.int_to_str[id] for id in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text
```

1. Convert each ID back to its token using `int_to_str` (the **inverse vocabulary**)
2. Join all tokens with spaces
3. Clean up spaces before punctuation marks (e.g. `word ,` → `word,`)

→ `[7, 0, 1, ...]` becomes `"The brown dog..."`

---

### The two vocabularies

| Attribute | Type | Direction | Built from |
|-----------|------|-----------|------------|
| `self.str_to_int` | `dict` | token → ID | `vocab` passed at init |
| `self.int_to_str` | `dict` | ID → token | inverted from `vocab` |

The **inverse vocabulary** is created automatically in `__init__`:
```python
self.int_to_str = {i: s for s, i in vocab.items()}
```

This symmetry is what makes encode/decode perfectly reversible — as long as all tokens in the input exist in the vocabulary.

---

## Key Concepts

- **Tokenization** — Splitting text into units (tokens) the model can work with. Tokens can be words, subwords, or individual characters depending on the tokenizer.
- **Vocabulary** — A fixed mapping from tokens to integer IDs. GPT-2 uses a vocabulary of ~50,000 tokens.
- **Token embeddings** — Dense vector representations of token IDs. These are *learned* during training, not fixed.
- **Decoder-only Transformer** — The architecture used by GPT-style models. It processes tokens left-to-right and predicts the next token.
