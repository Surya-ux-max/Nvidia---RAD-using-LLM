# Notebook 5: Multimodal Architectures and Fusion Techniques

Documentation for [05_multimodal.ipynb](file:///d:/Nvidia-RADU-LLM/05_multimodal.ipynb) from the **NVIDIA Deep Learning Institute (DLI): Rapid Application Development using Large Language Models**.

---

## 1. Executive Summary & Multimodal Paradigm Taxonomy

Notebook 5 transitions beyond text-only models into **multimodal generative AI systems** capable of processing and generating combinations of **Text**, **Images**, and **Audio**. It explores how heterogeneous data structures are encoded into compatible representation spaces and evaluates different fusion strategies: **Bi-Encoders (Late Fusion)**, **Cross-Encoders (Early Fusion)**, **Cross-Attention Decoders (Seq2Seq)**, and **Iteratively-Refined Diffusion Decoders**.

```
+----------------------------------------------------------------------------------------------------+
|                                  Multimodal Architectural Paradigms                                |
+------------------------------------+----------------------------------+----------------------------+
| Paradigm                           | Mechanism                        | Canonical Example          |
+------------------------------------+----------------------------------+----------------------------+
| 1. Late Fusion (Bi-Encoder)        | Dual encoders + shared projection| OpenAI CLIP                |
| 2. Early Fusion (Cross-Encoder)    | Joint concatenated intake pathway| VLM Vision-Language Models |
| 3. Cross-Attention Seq2Seq         | Visual/Audio Encoder + LLM Dec   | ViT-GPT2, Whisper          |
| 4. Iterative Diffusion Decoders    | Non-autoregressive latent denoise| Stable Diffusion XL        |
+------------------------------------+----------------------------------+----------------------------+
```

---

## 2. Core Technical Concepts & Modality Encoders

A **modality** defines the structural organization through which information is expressed:
- **Natural Language**: 1D discrete symbolic sequence with semantic contextual dependencies.
- **Audio Signals**: Continuous 1D time-series pressure waveforms or 2D time-frequency Mel-spectrograms.
- **Images**: 2D spatial pixel tensors ($H \times W \times C$) capturing micro fine-detail and macro compositional context.

### 2.1 Unifying Modality Encoders to 768-Dimensional Representations

The notebook standardizes three distinct data modalities into continuous $\mathbb{R}^{768}$ vector representations:

```python
# Modality Encoding Summary:
# 1. Text:      BERT-Base    -> (batch, seq_len=7, hidden_size=768)
# 2. Audio:     Wav2Vec 2.0  -> (batch, time_slices=200, hidden_size=768)
# 3. Vision:    ViT-Base     -> (batch, patches=197, hidden_size=768)
```

| Modality | Encoder Model | Input Format | Internal Tokenization / Patching | Output Tensor Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Text** | `bert-base-uncased` | Raw string | WordPiece subword tokens | `torch.Size([8, 7, 768])` |
| **Audio** | `facebook/wav2vec2-base-960h` | 16 kHz `.wav` waveform | Convolutions over ~0.02s time windows | `torch.Size([1, 200, 768])` |
| **Images** | `google/vit-base-patch16-224` | $224 \times 224 \times 3$ RGB | $16 \times 16$ pixel patches ($14 \times 14 = 196 + 1$ `[CLS]`) | `torch.Size([4, 197, 768])` |

---

## 3. Section-by-Section Technical Walkthrough

### Part 5.2 & 5.3: Joint Projections & Contrastive Learning with CLIP

When encoders are trained in isolation, their 768-dimensional latent spaces share no semantic correlation. Comparing independently trained BERT text embeddings with ViT image embeddings produces near-random similarity scores.

#### Contrastive Alignment via OpenAI CLIP (`openai/clip-vit-base-patch32`):
- **Bi-Encoder Architecture**: Utilizes an independent text transformer and an independent vision transformer.
- **Contrastive Objective**: Maximizes cosine similarity for matching (image, text) pairs along the matrix diagonal while penalizing off-diagonal negative pairs.
- **Normalized Dot-Product**:
  $$\text{Similarity}(T, I) = \text{softmax}\left(\frac{T_{\text{norm}} \cdot I_{\text{norm}}^T}{\tau}\right)$$

#### Implementation:
```python
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_captions = ("Cat with paint", "rock statue", "frogs on leaf", "jellyfish")
text_dialogue = ("Cats don't usually", 'like water', 'but this one likes paint', 'quite a lot')
img_files = ["paint-cat", "rock-head", "tree-frog", "two-jelly"]

inputs_text = processor(text=text_captions + text_dialogue, return_tensors="pt", padding=True)
inputs_images = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**inputs_text)
    image_features = model.get_image_features(**inputs_images)

# Normalized similarity matrix for captions vs images
sim_mtx = (text_features[:4] @ image_features.T).softmax(dim=-1)
```

#### Observations:
- **Descriptive Captions**: Accurately align with their respective images along the diagonal (e.g., `"Cat with paint"` strongly matches `paint-cat.jpg`).
- **Conversational Dialogue**: Generalizes poorly when text lacks descriptive visual grounding, illustrating that bi-encoders reflect their specific contrastive training distribution.

---

### Part 5.4: Combining Multimodal Encoders with Decoders

For generative outputs (textual descriptions, transcriptions), simple late-fusion similarity comparison is insufficient. A domain-specific decoder must condition its step-by-step token generation on the encoder latents via **cross-attention**.

#### 1. Image Captioning with ViT + GPT-2 (`nlpconnect/vit-gpt2-image-captioning`)
- **Visual Encoder (ViT)**: Ingests $224 \times 224$ images and produces patch latents.
- **Language Decoder (GPT-2)**: Generates descriptive captions autoregressively, querying visual features through cross-attention layers.
- **Prompted Generation**: Conditioning the decoder with prefix tokens (`"I love"`) forces stylistic control:

```python
from transformers import pipeline

vit_pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device="cuda")
captions = [vit_pipe(image) for image in images]
```

**Expected Output:**
- `paint-cat`: `"a cat with a purple and green eyes"`
- `rock-head`: `"a statue of a bird on a white surface"`
- `tree-frog`: `"a small yellow and black bird sitting on a plant"`
- `two-jelly`: `"a black and white photo of a blue and white kite"`

---

#### 2. Speech-to-Text Transcription with Whisper (`openai/whisper-large-v3`)
- **Audio Encoder**: Ingests 80-channel log-magnitude Mel-spectrograms extracted from 30-second audio windows.
- **Text Decoder**: Autoregressively generates tokenized transcriptions with punctuation and casing.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")

audio, _ = torchaudio.load("audio-files/paint-cat.wav")
inputs = processor(audio.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).to("cuda")

generated_ids = model.generate(inputs.input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(transcription)
```

**Expected Transcription:**
```text
"Cats don't usually like water, but this one seems to like paint quite a lot."
```

---

#### 3. Vision-Language Models (VLMs) Architecture Overview
Modern VLMs (e.g., LLaVA, BLIP-2) adopt a **decoder-only language backbone**:
1. A visual encoder (such as ViT or SigLIP) encodes image patches.
2. A lightweight **linear projection / MLP adapter** projects image features into the exact embedding dimension of the LLM.
3. Projected visual tokens are prepended directly to text tokens, allowing the unified causal decoder to reason over both modalities in a single sequence.

---

### Part 5.5 & 5.6: Non-Autoregressive Diffusion Decoders

#### Why Autoregression Fails for High-Resolution Images:
Autoregressive models generate tokens raster-scan style (pixel-by-pixel or patch-by-patch). Because images possess complex 2D bidirectional spatial coherence, autoregression suffers from error accumulation, blurriness, and lack of high-frequency detail preservation.

#### Denoising Diffusion Formulation (SDXL):
1. **Forward Process (Markov Chain)**: Gradually corrupts an image latent $z_0$ by adding Gaussian noise over $T$ timesteps until it becomes isotropic noise $z_T \sim \mathcal{N}(0, I)$.
2. **Reverse Process (Denoising Network)**: A UNet or Diffusion Transformer predicts the noise component $\epsilon_\theta(z_t, t, c)$ conditioned on text embedding $c$ (from CLIP / OpenCLIP).
3. **Latent Space Optimization**: Denoising occurs within a compressed latent space ($8\times$ downscaled) via a Variational Autoencoder (VAE), drastically lowering computational complexity.

```
+--------------------------------------------------------------------------------------------------+
|                                    SDXL Latent Diffusion Pipeline                                 |
+--------------------------------------------------------------------------------------------------+
 Text Prompt ---> [Text Encoders (CLIP)] ---> Conditioning Embedding (c)
                                                            |
 Initial Noise Latent (z_T) ---> [Iterative UNet Denoising Loop (Steps 1..N)] <--- Timestep (t)
                                                            |
                                               Cleaned Latent (z_0)
                                                            |
                                                [VAE Decoder]
                                                            |
                                                    Final Image
```

---

### Inspecting the Denoising Trajectory via Callback Hooks

The notebook taps into the internal execution loop of `stabilityai/stable-diffusion-xl-base-1.0` using `callback_on_step_end` to decode and visualize latent states across 10 denoising steps:

```python
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

step_images = []

def save_step_image(pipeline, step, timestep, callback_kwargs):
    latents = callback_kwargs['latents']
    # Scale latents and decode via VAE
    latent_scale = 1 / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latents * latent_scale, return_dict=False)[0]
    step_images.append(image)
    return callback_kwargs

prompt = "An armored man in green cape, Dr. Doom as a god emperor"

pipe(
    prompt=prompt,
    num_inference_steps=10,
    guidance_scale=7.5,
    callback_on_step_end=save_step_image,
    callback_on_step_end_tensor_inputs=['latents']
)

# Plot progressive resolution across steps 1 through 10
fig, axes = plt.subplots(1, len(step_images), figsize=(20, 5))
for i, img in enumerate(step_images):
    axes[i].imshow((img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
    axes[i].set_title(f"Step {i+1}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
```

#### Trajectory Analysis:
- **Steps 1–3**: Global color layout and macro contours emerge from noise.
- **Steps 4–7**: Semantic boundaries (armor shape, cape, posture) form.
- **Steps 8–10**: High-frequency textures, specular highlights, and edge sharpness are finalized.

---

### Part 5.7: Resource Teardown
```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```
- Restarts the IPython kernel process to immediately release GPU VRAM occupied by heavy SDXL diffusion pipelines, Whisper weights, and ViT checkpoints.

---

## 4. Key Takeaways & Architectural Decisions

1. **Late vs. Early Fusion**: Bi-encoders (CLIP) offer extreme speed for zero-shot search and retrieval ($O(N + M)$ encoding compute), while cross-attention decoders allow deeper token-level inter-modal reasoning ($O(N \times M)$ generative compute).
2. **Sequential vs. Iterative Decoders**: Autoregression remains state-of-the-art for sequence outputs (text, code, transcribed audio), whereas iterative denoising diffusion decoders excel for high-dimensional spatial outputs (images, audio waveforms, 3D meshes).
3. **Cross-Attention as a Universal Bridge**: Any modal encoder (audio, visual, tactile) can condition an LLM text decoder provided keys and values are projected into compatible hidden dimensions.
