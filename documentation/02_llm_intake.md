# Notebook 2: LLM Architecture Intuitions & Intake Mechanisms

Documentation for [02_llm_intake.ipynb](file:///d:/Nvidia-RADU-LLM/02_llm_intake.ipynb) from the **NVIDIA Deep Learning Institute (DLI): Rapid Application Development using Large Language Models**.

---

## 1. Executive Summary & Core Concepts

Notebook 2 transitions from high-level Hugging Face pipeline abstractions to the foundational mechanics of transformer-based models (specifically `bert-base-uncased`). It unpacks how natural language text is transformed into numerical tensors, how semantic and positional geometries are represented, and how self-attention bridges the gap between individual tokens and sequence-level comprehension.

### Core Architectural Pillars:
1. **Intake / Tokenization**: Converting unstructured text into discrete integer identifiers (`input_ids`), segment identifiers (`token_type_ids`), and attention constraints (`attention_mask`).
2. **Tri-Partite Embedding Space**: Combining token vocabulary embeddings, positional embeddings, and segment/type embeddings via vector addition rather than concatenation.
3. **Similarity Spaces**: Measuring vector similarity via Cosine Similarity, Scaled Dot-Product, and Softmax-scaled attention.
4. **Self-Attention Mechanics ($Q, K, V$)**: Enabling bidirectional context where sequence elements dynamically attend to other elements using parameterized query, key, and value matrices.

---

## 2. Git Push Readiness Audit for `02_llm_intake.ipynb`

Prior to committing and pushing to remote repositories, the notebook was audited across critical hygiene metrics:

| Check Item | Status | Verification Detail |
| :--- | :---: | :--- |
| **API Keys & Secrets** | ✅ Clean | No private API keys (OpenAI, Hugging Face tokens, NGC keys) are present. |
| **Cell Execution Outputs** | ✅ Clean | All cell outputs are stripped (`"outputs": []`, `"execution_count": null`), preventing bloated binary metadata or leaked system traces. |
| **Sensitive System Paths** | ✅ Clean | No local absolute filesystem paths containing usernames or environment secrets. |
| **JSON Schema Integrity** | ✅ Valid | Adheres to Jupyter Notebook v4.4 specification; ready for Git tracking. |
| **File Footprint** | ✅ Minimal | Clean file size (~30 KB), ideal for version control. |

---

## 3. Section-by-Section Technical Walkthrough

### Part 2.1: Model Intake & Tokenization

Tokenization segments arbitrary character streams into discrete vocabulary entries (words, punctuation, or subwords).

#### Code:
```python
from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Testing out the tokenizer
msg = "Hello world!"
print("Tokenize:", unmasker.tokenizer.tokenize(msg))      # See token boundaries
print("Encoding:", x := unmasker.tokenizer.encode(msg))   # See special tokens at ends
print("Decoding:", x := unmasker.tokenizer.decode(x))     # See decoding

# Specifying multiple sentences
unmasker.tokenizer("Hello world!", "Have a great day!")
```

#### Explanation:
- `pipeline('fill-mask', model='bert-base-uncased')`: Loads BERT's Masked Language Model (MLM) head with WordPiece tokenization.
- `tokenize()`: Splits `"Hello world!"` into individual token strings `['hello', 'world', '!']`.
- `encode()`: Appends special boundary tokens (`[CLS]` at index 101, `[SEP]` at index 102) and returns integer IDs: `[101, 7592, 2088, 999, 102]`.
- `decode()`: Reconstructs the canonical string representation from IDs.
- Two-sentence intake produces `token_type_ids` (`0` for first sentence, `1` for second sentence), enabling cross-sentence classification tasks.

#### Expected Output:
```text
Tokenize: ['hello', 'world', '!']
Encoding: [101, 7592, 2088, 999, 102]
Decoding: [CLS] hello world ! [SEP]
{'input_ids': [101, 7592, 2088, 999, 102, 2031, 1037, 2307, 2154, 999, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

---

### Part 2.2: Capturing Token Semantics & Embedding Architecture

BERT maps discrete token IDs into dense continuous vector spaces ($\mathbb{R}^{768}$).

```
Embedding Composition Formula:
Embedding = WordEmbed[token] + PosEmbed[position] + TokenTypeEmbed[sentence_id]
Followed by: LayerNorm(Embedding) -> Dropout(Embedding)
```

#### 1. Inspecting BERT Embeddings Hierarchy
```python
model = unmasker.model
model.bert.embeddings
```
**Architecture Breakdown:**
- `word_embeddings`: `Embedding(30522, 768)` (30,522 vocabulary items mapped to 768 dimensions)
- `position_embeddings`: `Embedding(512, 768)` (Context window of up to 512 tokens)
- `token_type_embeddings`: `Embedding(2, 768)` (Distinguishes sentence A from sentence B)
- `LayerNorm`: Normalizes across feature dimensions ($d=768$) to stabilize gradient dynamics.
- `Dropout`: Rate of 0.1 to prevent co-adaptation of features.

---

#### 2. Word Embedding Extraction
```python
import torch

tokenizer = unmasker.tokenizer

string = "Hello World From Me, my cat and my dog!"
tokens = [tokenizer.convert_ids_to_tokens(x) for x in tokenizer.encode(string)]
token_ids = torch.tensor(tokenizer(string)['input_ids']).cuda()
embeddings = model.bert.embeddings.word_embeddings(token_ids).cpu()
print(embeddings.shape)
```
- **Output Tensor Shape**: `torch.Size([13, 768])` representing 13 tokens (including `[CLS]` and `[SEP]`), each mapped to 768 dimensions.

---

#### 3. Embedding Similarity Metrics

To quantify how tokens relate to each other, three similarity formulations are evaluated:

| Metric | Mathematical Formula | Characteristics |
| :--- | :--- | :--- |
| **Cosine Similarity** | $\frac{x_1 \cdot x_2^T}{\|x_1\|_2 \|x_2\|_2}$ | Measures angular alignment between $[-1, 1]$; ignores vector magnitude. |
| **Scaled Dot-Product** | $\frac{x_1 \cdot x_2^T}{\sqrt{d_k}}$ | Preserves magnitude; scaling factor $\sqrt{d_k}$ prevents vanishing gradients in softmax. |
| **Softmax Similarity** | $\text{softmax}\left(\frac{x_1 \cdot x_2^T}{\sqrt{d_k}}\right)$ | Asymmetric, normalized probability distribution ($0$ to $1$) where row sums equal $1.0$. Represents foundational attention weight distribution. |

```python
def cosine_similarity(x1, x2):
    x1_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    x2_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    return x1_norm @ x2_norm.T

def scaled_dp_similarity(x1, x2):
    dot_product = x1 @ x2.T
    d = torch.sqrt(torch.tensor(x1.shape[-1], dtype=torch.float32))
    return dot_product / d

def softmax_similarity(x1, x2):
    out = scaled_dp_similarity(x1, x2)
    return torch.softmax(out, dim=1)
```

---

#### 4. Positional Embeddings vs Word Embeddings
Unlike recurrent models (RNNs/LSTMs) that ingest data step-by-step, Transformers process all tokens concurrently. Absolute positional embeddings inject spatial order:
```python
pos_options = torch.arange(len(tokens)).cuda()
pos_embeddings = model.bert.embeddings.position_embeddings(pos_options).cpu()
plot_mtx(cosine_similarity(pos_embeddings, pos_embeddings), 'Cosine Sim', tokens)
```
- **Observation**: Positional embeddings exhibit smooth, monotonic diagonal decay—tokens closer in the sequence share higher similarity than distant tokens.

---

### Part 2.3: Sequence-Level Reasoning & Self-Attention

#### Why Token-by-Token and Dense Layers Fail:
- **Token-by-Token reasoning**: Misses context (e.g., distinguishing "apple" the fruit vs "Apple" the company).
- **Global Dense Layer ($N \times d \to N \times d$)**: Parameters explode with variable sequence lengths and cannot generalize across arbitrary contexts.
- **The Transformer Solution**: Preserve token-level representations while introducing **Self-Attention** bottlenecks where tokens exchange semantic context.

#### The Self-Attention Equation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Query ($Q$)**: What the current token is looking for.
- **Key ($K$)**: What features the current token offers.
- **Value ($V$)**: The actual content payload passed downstream.
- **Multi-Head Attention (BERT-Base)**:
  - 12 independent attention heads.
  - Dimension per head: $d_k = 768 / 12 = 64$.
  - Outputs of all 12 heads are concatenated back to dimension $768$.

---

### Part 2.4: Inspecting BERT's Encoder Layer & Attention Tensors

#### Code:
```python
import torch

string = "Hello Mr. Bert! How is it [MASK]?"
input_tensors = unmasker.preprocess(string)
input_ids = unmasker.tokenizer.encode(string)
input_tokens = unmasker.tokenizer.convert_ids_to_tokens(input_ids)

embeddings = unmasker.model.bert.embeddings(input_tensors['input_ids'].cuda())
x = unmasker.model.bert.encoder(embeddings, input_tensors['attention_mask'].cuda(), output_attentions=True)

print("From encoder.forward():")
for k, v in x.items():
    v = v if hasattr(v, "shape") else torch.stack(v)
    print(f" > '{k}' : {v.shape}")
```

#### Output Analysis:
```text
From encoder.forward():
 > 'last_hidden_state' : torch.Size([1, 11, 768])
 > 'attentions' : torch.Size([12, 1, 12, 11, 11])
```
- `last_hidden_state`: Tensor of shape `[batch_size=1, seq_len=11, hidden_dim=768]`.
- `attentions`: 12 encoder layers, each with `[batch_size=1, num_heads=12, seq_len=11, seq_len=11]` attention weight matrices.

---

### Part 2.5: Interactive Attention Visualization with BertViz

```python
from bertviz import model_view

model_view(x['attentions'], input_tokens)
```
- Visualizes interactive multi-head attention maps across all 12 layers.
- **Key Pattern**: Special tokens `[CLS]` and `[SEP]` frequently act as attention sinks, aggregating global sentence representations for downstream classification tasks.

---

### Part 2.6: Resource Teardown
```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```
- Restarts the IPython kernel process to immediately deallocate GPU memory allocated by PyTorch tensors and BERT model weights.

---

## 4. Key Takeaways & Best Practices
1. **Additive Embeddings**: Word identity, sequence position, and segment type are fused through vector addition before entering the first transformer block.
2. **Attention Scaling**: The factor $\frac{1}{\sqrt{d_k}}$ is essential to keep dot-product magnitudes in regions where softmax gradients do not vanish.
3. **Multi-Head Specialization**: Different attention heads specialize in distinct linguistic structures (e.g., direct syntactic dependencies, coreference resolution, delimiter tracking).
