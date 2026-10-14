# Notebook 3: LLM Encoder Tasks & Task-Specific Heads

Documentation for [03_encoder_task.ipynb](file:///d:/Nvidia-RADU-LLM/03_encoder_task.ipynb) from the **NVIDIA Deep Learning Institute (DLI): Rapid Application Development using Large Language Models**.

---

## 1. Overview & Core Concepts

Notebook 3 investigates how pretrained transformer encoder backbones (such as BERT and RoBERTa) are adapted for diverse downstream natural language processing tasks. By swapping lightweight **task-specific classification heads** on top of frozen or fine-tuned backbone latents, the same encoder representations solve token-level, span-level, and passage-level problems without altering the underlying self-attention architecture.

### Architectural Paradigms Covered:
```
                      +---------------------------------------+
                      |       Input Sequence of Tokens        |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Pretrained Transformer Backbone     |
                      |   (BERT / RoBERTa contextual latents) |
                      +---------------------------------------+
                                          |
       +----------------------------------+----------------------------------+
       |                                  |                                  |
       v                                  v                                  v
+-----------------------+      +-----------------------+      +-----------------------+
|  Token Classification |      |   Range / Span Output  |      | Sequence Classification|
|  (e.g., Masked LM)    |      | (Question Answering)  |      | (Sentiment / Emotions)|
|  dim: 768 -> 30,522   |      | dim: 768 -> 2 (start/end)    | dim: 768 -> C classes |
|  Outputs: Per-Token   |      |  Outputs: Substring   |      |  Outputs: Single Label|
+-----------------------+      +-----------------------+      +-----------------------+
```

---

## 2. Section-by-Section Technical Walkthrough

### Part 3.1: The Token Prediction Task Head (Masked Language Modeling)

In Masked Language Modeling (MLM), each token latent vector is projected back into vocabulary space ($30,522$ dimensions) to recover hidden or masked tokens (`[MASK]`).

#### Pipeline Architecture:
- Backbone: `bert-base-uncased` ($L=12, H=768, A=12$)
- Task Head: `BertOnlyMLMHead`
  - `BertPredictionHeadTransform`: `Linear(768, 768)` $\to$ `GELUActivation()` $\to$ `LayerNorm(768)`
  - `decoder`: `Linear(768, 30522)`

#### Dissected Pipeline Implementation:
```python
from transformers import BertTokenizer, BertModel, FillMaskPipeline, AutoModelForMaskedLM
import torch

class MyFillMaskModel(FillMaskPipeline):
    def __init__(self):
        super().__init__(
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
            model=AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        )

    def __call__(self, string):
        # 1. Preprocess: Tokenize string and locate [MASK] token (ID = 103)
        input_tensors = self.preprocess(string)
        mask_idx = (input_tensors['input_ids'] == 103).nonzero()[0][1].item()

        # 2. Forward Pass: Compute embeddings, encoder latents, and projection logits
        inputs_1 = {'input_ids': input_tensors['input_ids'].cuda()}
        inputs_2 = {'attention_mask': input_tensors['attention_mask'].bool().cuda()}
        embed_out = self.model.bert.embeddings.forward(**inputs_1)
        bert_out = self.model.bert.encoder.forward(embed_out, **inputs_2)['last_hidden_state']
        y = self.model.cls.forward(bert_out)

        # 3. Softmax probability distribution over the vocabulary
        pdfs = torch.softmax(y[0], -1)

        # 4. Extract top-k candidate words for the masked index
        mask_top_probs = torch.topk(pdfs[mask_idx], 5)
        mask_best_words = [self.tokenizer.decode(idx) for idx in mask_top_probs.indices]
        return self.postprocess({**input_tensors, 'logits': y})

unmasker = MyFillMaskModel()
result = unmasker("Say [MASK]!")
```

#### Output Shapes & Tensor Flow:
```text
Statistics From Forward Pass:
> Input Into BERT Encoder: torch.Size([1, 5, 768])
> Input Into Classifier:   torch.Size([1, 5, 768])
> Output From Classifier:  torch.Size([1, 5, 30522])

Statistics From Postprocessing (Top 5):
> Candidates: ['something', 'nothing', 'so', 'yes', 'it']
```

---

### Part 3.2: Token-Level Range Prediction (Extractive Question Answering)

Rather than predicting vocabulary items, range-prediction heads determine where an answer span begins and ends within the provided context.

#### Key Mechanics:
- **Model**: `deepset/roberta-base-squad2` fine-tuned on the SQuAD 2.0 dataset.
- **Head Architecture**: `qa_outputs = Linear(in_features=768, out_features=2)`
  - Outputs 2 logits per token position: `start_logits` and `end_logits`.
  - The answer is bounded by maximizing $\text{score} = \text{start\_logits}[i] + \text{end\_logits}[j]$ subject to $i \le j$.
- **Hallucination Mitigation**: Constraining the model to extract verbatim substrings from context guarantees factual consistency and eliminates ungrounded fabrications.

#### Code:
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
result = nlp(**QA_input)
```

#### Expected Output:
```json
{
  "score": 0.2117,
  "start": 59,
  "end": 84,
  "answer": "gives freedom to the user"
}
```

---

### Part 3.3: Sequence-Level Classification (Sentiment & Emotion Analysis)

For passage-level classification, the contextual representation of the first token (index `0`: `[CLS]` in BERT, `<s>` in RoBERTa) aggregates full-sequence context via self-attention.

#### Architecture (`RobertaClassificationHead`):
```python
# emo_model.model.classifier
RobertaClassificationHead(
  (dense): Linear(in_features=768, out_features=768, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (out_proj): Linear(in_features=768, out_features=28, bias=True)
)
```

#### Code:
```python
from transformers import pipeline

emo_model = pipeline('sentiment-analysis', 'SamLowe/roberta-base-go_emotions')

print(emo_model("I love my old pillow?"))
print(emo_model("Why is it that every plant I touch dies within a few days?"))
print(emo_model("I'm so conflicted about these new instructions..."))
```

#### Expected Output:
```text
[{'label': 'love', 'score': 0.6621}]
[{'label': 'sadness', 'score': 0.6277}]
[{'label': 'confusion', 'score': 0.7412}]
```

---

### Part 3.4: Zero-Shot Classification via Natural Language Inference (NLI)

Zero-shot classification classifies passages into arbitrary, user-defined categories without requiring labeled examples or task-specific re-training.

#### Theoretical Foundation:
- Uses a model trained on Multi-Genre Natural Language Inference (**MultiNLI**), such as `facebook/bart-large-mnli`.
- Formulates classification as premise-hypothesis entailment:
  - **Premise ($P$)**: The text to classify.
  - **Hypothesis ($H$)**: A template string: `"This example is {category}."`
  - The model queries entailment probabilities for each candidate category independently and applies softmax across classes.

#### Implementation (Task 1 Solution):
```python
from transformers import pipeline

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence = "I am designing a distributed inference pipeline using TensorRT-LLM and Triton Inference Server."
candidate_labels = ["artificial intelligence", "culinary arts", "aerospace engineering", "finance"]

output = zero_shot_classifier(sequence, candidate_labels)
print(output)
```

#### Expected Output:
```json
{
  "sequence": "I am designing a distributed inference pipeline using TensorRT-LLM and Triton Inference Server.",
  "labels": ["artificial intelligence", "finance", "aerospace engineering", "culinary arts"],
  "scores": [0.9315, 0.0321, 0.0245, 0.0119]
}
```

---

### Part 3.5: Resource Teardown & GPU VRAM Cleanup
```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```
- Completely halts the active PyTorch CUDA session to reclaim GPU VRAM before proceeding to the next exercise.

---

## 3. Comparison of Encoder Task Heads

| Task Type | Pipeline Type | Target Head Architecture | Output Dimension | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **Masked Language Modeling** | `fill-mask` | `BertOnlyMLMHead` (`Linear` + `GELU` + `LayerNorm` + `Linear`) | $N \times 30,522$ | Predicts probability distribution across entire vocabulary for masked tokens. |
| **Question Answering** | `question-answering` | `Linear(768, 2)` | $N \times 2$ | Generates start and end span pointers; guarantees verbatim factual answers. |
| **Sequence Classification** | `sentiment-analysis` | `RobertaClassificationHead` (`Linear` + `Dropout` + `Linear`) | $1 \times C$ | Routes first sequence token (`[CLS]` / `<s>`) to classify entire text passage. |
| **Zero-Shot Classification** | `zero-shot-classification` | `BartForSequenceClassification` (MultiNLI head) | $K \times 3$ (Entailment / Neutral / Contradiction) | Re-frames classification as textual entailment for arbitrary runtime labels. |
