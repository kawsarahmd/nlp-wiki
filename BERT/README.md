
## 1. BERT (2018)

**Name + Year + Context**

* **BERT** = *Bidirectional Encoder Representations from Transformers*
* **Year:** 2018
* **Context:** First widely adopted transformer encoder pre-trained with **masked language modeling (MLM)** and **next sentence prediction (NSP)**; kicked off the “pre-train then fine-tune” era for NLP.

**What BERT does (baseline everyone else extends)**

* Uses a **Transformer encoder**.
* Pre-training objective:

  * **MLM:** Randomly mask ~15% of tokens and ask the model to predict them.
  * **NSP:** Given two sentences, predict whether sentence B actually follows sentence A.
* Bidirectional: It looks **both left and right context** when predicting a masked token.

**What problem BERT solves**

Before BERT, many models (like GPT-1 or ELMo variants) weren’t fully bidirectional or were task-specific.
BERT gives a **general-purpose language understanding model** you can fine-tune for many tasks with minimal task-specific architecture changes.

**Toy example (MLM)**

Sentence:

> “The cat sat on the **mat**.”

Masked input:

> “The cat sat on the `[MASK]`.”

BERT uses **all** surrounding words (“The cat sat on the …”) to guess “mat.” That’s it. But this simple idea, done at massive scale, gives rich word representations.

**Where it fits historically**

* BERT is the **starting point**.
* Almost every later model in this list is a “BERT, but…” variation.

**One-sentence summary**

> BERT introduced large-scale bidirectional pre-training (MLM + NSP), giving a universal language understanding backbone that revolutionized NLP.

---

## 2. RoBERTa (2019)

**Name + Year + Context**

* **RoBERTa** = *Robustly Optimized BERT Pretraining Approach*
* **Year:** 2019
* **Context:** Facebook AI revisited BERT and asked: “What if we just trained it **better and longer** with more data and no NSP?”

**What changed vs original BERT**

* **Removed NSP** objective entirely.
* **More data**, **larger batches**, **longer training**.
* **Dynamic masking** (masking pattern changes each epoch instead of being fixed once).
* Same model architecture as BERT, but with **much better training recipe**.

**What problem it solves**

Shows that **training recipe matters a lot**. Much of BERT’s performance is about *how* you train, not just the architecture.

**Toy example (dynamic masking)**

Text:

> “The cat sat on the mat and looked at the dog.”

Epoch 1 mask:

> “The cat `[MASK]` on the mat and looked at the dog.”

Epoch 2 mask:

> “The cat sat on the `[MASK]` and looked at the dog.”

Epoch 3 mask:

> “The `[MASK]` sat on the mat and looked at the `[MASK]`.”

RoBERTa keeps seeing different masked versions of the same sentence, forcing it to learn **richer representations**.

**Where it fits historically**

* Comes **right after BERT** (2019).
* Think of it as “BERT v1.5 with a really good training schedule.”

**One-sentence summary**

> RoBERTa shows that with more data, longer training, dynamic masking, and no NSP, you can significantly improve BERT without changing the architecture.

---

## 3. DistilBERT (2019)

**Name + Year + Context**

* **DistilBERT** = *Distilled BERT*
* **Year:** 2019
* **Context:** Hugging Face model that uses **knowledge distillation** to produce a smaller, faster BERT.

**What changed vs original BERT**

* Fewer layers (e.g., 6 instead of 12).
* Trained to **mimic BERT’s behavior** using teacher–student training (distillation).
* 40–60% smaller and faster with ~95% of BERT’s performance.

**What problem it solves**

BERT is **heavy and slow** for deployment (especially on CPUs and mobile). DistilBERT keeps most of the accuracy while being much lighter.

**Toy example (distillation intuition)**

* Teacher (BERT) sees:

  > “The cat sat on the `[MASK]`.”
  > BERT’s prediction:

  * “mat”: 0.85
  * “sofa”: 0.05
  * “floor”: 0.03
* Student (DistilBERT) is trained to match **these probability distributions**, not just the correct answer.

So DistilBERT doesn’t just learn “mat is correct”; it learns **how confident BERT is** about alternatives, inheriting its “dark knowledge.”

**Where it fits historically**

* After BERT, alongside RoBERTa (2019).
* First popular **compressed BERT**.

**One-sentence summary**

> DistilBERT compresses BERT using knowledge distillation to create a smaller, faster model that preserves most of BERT’s performance.

---

## 4. ALBERT (2019)

**Name + Year + Context**

* **ALBERT** = *A Lite BERT*
* **Year:** 2019
* **Context:** Google & TOYOTA Technological Institute; aggressively reduces parameters while keeping or improving performance.

**What changed vs original BERT**

* **Factorized embeddings:** Split large embedding matrix into two smaller matrices (e.g., vocab → 128 → hidden size).
* **Cross-layer parameter sharing:** Same weights are reused across multiple transformer layers.
* Replaced NSP with **Sentence Order Prediction (SOP)** to better capture sentence-level coherence.

**What problem it solves**

BERT has **huge embedding matrices** and lots of repeated parameters. ALBERT makes models that are **much smaller in memory** but still powerful.

**Toy example (sentence order prediction)**

Given pair:

1. “He put on his shoes.”
   “Then he went outside.”
2. “Then he went outside.”
   “He put on his shoes.”

BERT’s NSP: “Do these two sentences come from the same document?”
ALBERT’s SOP: “Are these sentences in the **correct order**?”

This better trains **inter-sentence coherence**.

**Where it fits historically**

* After BERT & RoBERTa (2019).
* Parallel to efficiency-oriented models like DistilBERT, but with more **parameter-sharing tricks**.

**One-sentence summary**

> ALBERT reduces memory and parameters via factorized embeddings and cross-layer sharing while improving sentence-level modeling with sentence-order prediction.

---

## 5. SpanBERT (2019)

**Name + Year + Context**

* **SpanBERT**
* **Year:** 2019
* **Context:** Facebook AI; focuses on better **span (phrase) representations** rather than just single tokens.

**What changed vs original BERT**

* Instead of randomly masking individual tokens, it masks **contiguous spans** of text.
* Objective: predict **the whole span** from its context (span-boundary objectives).
* No NSP; more emphasis on span-level pretraining.

**What problem it solves**

Many tasks (QA, coreference, NER) involve **phrases or spans**, not isolated words. SpanBERT gives better representations for those by training directly on spans.

**Toy example (span masking)**

Text:

> “The **quick brown fox** jumps over the lazy dog.”

SpanBERT masks the entire phrase:

> “The `[MASK][MASK][MASK]` jumps over the lazy dog.”

The model must reconstruct **“quick brown fox”** as a unit, not just predict “quick”, then “brown”, then “fox” separately.
This forces it to think in terms of **phrases**.

**Where it fits historically**

* 2019, shortly after BERT and around the same time as RoBERTa.
* Branch focused on **span-level tasks**.

**One-sentence summary**

> SpanBERT extends BERT’s masking from tokens to spans, yielding stronger representations for phrase-level tasks like QA and coreference.

---

## 6. ELECTRA (2020)

**Name + Year + Context**

* **ELECTRA** = *Efficiently Learning an Encoder that Classifies Token Replacements Accurately*
* **Year:** 2020
* **Context:** Google; aimed at making pretraining **more sample-efficient** than MLM.

**What changed vs original BERT**

* Replaces MLM with **Replaced Token Detection (RTD)**:

  * Small **generator** replaces some tokens with plausible alternatives.
  * Big **discriminator** is trained to decide: “Is this token real or replaced?”
* No need to predict only masked tokens; model learns from **every token**.

**What problem it solves**

MLM wastes computation: only 15% of tokens contribute to loss. ELECTRA learns from **all tokens**, making pretraining **faster and more efficient**.

**Toy example (RTD)**

Original sentence:

> “The cat sat on the mat.”

Generator output:

> “The cat sat on the sofa.”

Now the discriminator (ELECTRA encoder) gets:

> “The cat sat on the sofa.”

For each token, it predicts **REAL or FAKE**:

* “The” → REAL
* “cat” → REAL
* “sat” → REAL
* “on” → REAL
* “the” → REAL
* “sofa” → FAKE

This is more like **binary classification per token** than masked prediction.

**Where it fits historically**

* Comes after BERT, RoBERTa, ALBERT, SpanBERT (around 2020).
* Represents a **new pretraining objective family**.

**One-sentence summary**

> ELECTRA replaces masked language modeling with replaced-token detection, learning from all tokens and making pretraining much more sample-efficient.

---

## 7. DeBERTa (v1, v2, v3 – summarized)

**Name + Year + Context**

* **DeBERTa** = *Decoding-enhanced BERT with Disentangled Attention*
* **Year:** Around 2021 for v1; subsequent improvements for v2 and v3 by Microsoft.
* **Context:** Designed to improve attention mechanisms and position encoding.

**What changed vs original BERT**

Core DeBERTa ideas (applies across v1–v3):

1. **Disentangled attention:**

   * Separates **content embeddings** (what the token is) from **position embeddings** (where the token is).
   * Attention scores consider both content–content and content–position interactions more explicitly.
2. **Improved position encoding:**

   * Uses relative position encodings more effectively than vanilla BERT.

Version-level:

* **DeBERTa v1:** Introduced disentangled attention + enhanced mask decoder.
* **DeBERTa v2:** Improved training efficiency and larger-scale pretraining.
* **DeBERTa v3:** Uses ELECTRA-style **replaced-token detection objective** with DeBERTa architecture.

**What problem it solves**

BERT’s **absolute positional embeddings** and entangled content/position representations can limit generalization to longer or shifted sequences. DeBERTa gives **richer and more flexible** attention.

**Toy example (disentangled attention intuition)**

Sentence:

> “The cat chased the mouse.”

Consider attention between “chased” and other tokens:

* BERT: attention mixes *what* and *where* into a single vector and score.
* DeBERTa:

  * Content: “chased” vs “cat” / “mouse” (semantic relation).
  * Position: distance and order (e.g., “cat” is before “chased,” “mouse” is after).

Attention score = f(semantic similarity) + f(positional relation).
So DeBERTa can better model relations like **subject vs object** based on both meaning and position.

**Where it fits historically**

* After ELECTRA (around 2021+).
* Combines architectural improvements (disentangled attention) and newer objectives (v3 with RTD).

**One-sentence summary**

> DeBERTa refines BERT’s attention with disentangled content and position representations, and in later versions pairs this with efficient objectives like ELECTRA’s RTD.

---

## 8. Domain-Adaptive BERTs

*(BioBERT, ClinicalBERT, BERTweet, SciBERT)*

**General pattern**

* Start with BERT (or similar).
* Continue pretraining on **domain-specific corpora** (biomedical text, clinical notes, tweets, scientific papers, etc.).
* Sometimes adjust vocabulary/tokenizer as well.

---

### 8.1 BioBERT (2019)

**Name + Year + Context**

* **BioBERT**
* **Year:** 2019
* **Context:** Pretrained BERT further on biomedical corpora (PubMed, PMC).

**What changed vs original BERT**

* Same architecture and objectives (MLM + NSP).
* Extra pretraining on **biomedical text**.

**Problem it solves**

BERT’s understanding of terms like “BRCA1”, “angiogenesis”, “myocardial infarction” is weak. BioBERT **specializes** in biomedical language.

**Toy example**

Sentence:

> “The patient was treated with **trastuzumab**.”

* Vanilla BERT may treat “trastuzumab” as rare, poorly-understood token pieces.
* BioBERT has seen it many times, so it better understands relations like “trastuzumab ↔ cancer treatment.”

**One-sentence summary**

> BioBERT is BERT further trained on biomedical text, boosting performance on biomedical NLP tasks.

---

### 8.2 ClinicalBERT (2019)

**Name + Year + Context**

* **ClinicalBERT**
* **Year:** 2019
* **Context:** Specialized for **clinical notes** (e.g., MIMIC-III).

**What changed**

* Further pretraining BERT on **clinical narratives** (noisy, shorthand, abbreviations).

**Problem it solves**

Clinical text has lots of **shorthand and jargon**: “SOB”, “c/o”, “hx of DM”. ClinicalBERT learns this language.

**Toy example**

Text:

> “Pt c/o SOB, hx of HTN, DM2.”

Vanilla BERT: may struggle with abbreviations.
ClinicalBERT: learns “Pt” = patient, “SOB” = shortness of breath, “HTN” = hypertension, “DM2” = type 2 diabetes.

**One-sentence summary**

> ClinicalBERT adapts BERT to clinical note style, improving healthcare-related NLP tasks.

---

### 8.3 SciBERT (2019)

**Name + Year + Context**

* **SciBERT**
* **Year:** 2019
* **Context:** Pretrained on scientific articles (Semantic Scholar).

**What changed**

* New vocabulary based on scientific text.
* Trained from scratch or from BERT variants on **scientific corpora**.

**Problem it solves**

Scientific papers use specialized language, formulas, and domain-specific words (e.g., “convolutional”, “isomorphism”, “p-value”). SciBERT learns that environment.

**Toy example**

Sentence:

> “We propose a novel convolutional architecture for sequence modeling.”

SciBERT tokenizes and understands “convolutional”, “architecture”, “sequence modeling” in a **science-specific way**, leading to better performance on tasks like citation intent, scientific NER, etc.

**One-sentence summary**

> SciBERT is tailored to scientific literature, providing better representations for academic and technical NLP tasks.

---

### 8.4 BERTweet (2020)

**Name + Year + Context**

* **BERTweet**
* **Year:** 2020
* **Context:** BERT-like model trained on large-scale English **Twitter** data.

**What changed**

* New tokenizer and pretraining on noisy, informal Twitter text.

**Problem it solves**

Twitter language: hashtags, emojis, misspellings, slang, and short, fragmented sentences. BERTweet captures this style.

**Toy example**

Tweet:

> “omg that movie was lit 😂🔥”

Vanilla BERT: struggles with “omg”, “lit” (slang) and emojis.
BERTweet: has seen millions of such tweets and better understands sentiment and nuances.

**One-sentence summary**

> BERTweet adapts BERT to social media text, giving strong performance on tweet sentiment, hate speech detection, etc.

---

**Domain-adaptive one-liner (meta)**

> All these models show that taking BERT and **continuing pretraining on in-domain data** (biomed, clinical, scientific, tweets) massively boosts performance in that domain.

---

## 9. Multilingual BERTs (mBERT, XLM-R)

### 9.1 mBERT (2018)

**Name + Year + Context**

* **mBERT** = Multilingual BERT
* **Year:** 2018 (released with BERT)
* **Context:** Single BERT model trained on **104 languages** using Wikipedia.

**What changed vs original BERT**

* Shared **vocabulary and parameters** across many languages.
* Same MLM + NSP objectives.

**Problem it solves**

Gives **one model** that works on many languages, even those with limited labeled data.

**Toy example**

* English: “The cat sits on the mat.”
* Spanish: “El gato se sienta en la alfombra.”

mBERT uses a shared subword vocabulary (“cat/gato”, “mat/alfombra”) and can align representations so that similar words across languages have **similar embeddings**, enabling **zero-shot transfer** (train in English, test in Spanish).

**One-sentence summary**

> mBERT is a single BERT model trained on many languages, enabling multilingual and cross-lingual transfer.

---

### 9.2 XLM-R (2019)

**Name + Year + Context**

* **XLM-R** = *XLM-RoBERTa*
* **Year:** 2019
* **Context:** Facebook’s improved multilingual model, using RoBERTa-style training.

**What changed vs mBERT**

* Much **more multilingual data** (CommonCrawl).
* **RoBERTa training**: no NSP, more data, dynamic masking, longer training.
* Better cross-lingual performance than mBERT.

**Problem it solves**

mBERT struggles with low-resource languages and large-scale multilingual robustness. XLM-R is more **powerful and robust across languages**.

**Toy example**

Task: sentiment classification.
Train on English reviews: “This movie was amazing.”
Test on German reviews: “Dieser Film war fantastisch.”

XLM-R’s shared multilingual representations let it **transfer the sentiment knowledge** from English to German even without German labels.

**One-sentence summary**

> XLM-R is a RoBERTa-style multilingual model trained on massive CommonCrawl data, significantly improving cross-lingual performance over mBERT.

---

## 10. TinyBERT & MobileBERT

These focus on **on-device / low-resource** deployment.

---

### 10.1 TinyBERT (2020)

**Name + Year + Context**

* **TinyBERT**
* **Year:** 2020
* **Context:** A very small BERT variant using **two-stage knowledge distillation**.

**What changed vs original BERT**

* Fewer layers and smaller hidden sizes.
* Distillation both at **pretraining stage** and **task-specific stage**.
* Carefully designed to keep important behaviors while being tiny.

**Problem it solves**

When you need BERT-level performance on **small devices** (phones, edge devices), TinyBERT offers a **heavily compressed** alternative.

**Toy example**

Teacher (BERT) fine-tuned on sentiment.
Student (TinyBERT) learns:

* From BERT’s **intermediate hidden states** (matching layer outputs).
* From BERT’s **final predictions**.

So TinyBERT is like “BERT compressed twice”: once on generic language, once on each specific task.

**One-sentence summary**

> TinyBERT is a heavily compressed, distillation-based mini-BERT designed for efficient deployment with minimal performance loss.

---

### 10.2 MobileBERT (2020)

**Name + Year + Context**

* **MobileBERT**
* **Year:** 2020
* **Context:** Google’s architecture-optimized BERT variant specifically for **mobile/edge devices**.

**What changed vs original BERT**

* Uses **inverted bottleneck** architecture (small input/output dimension, larger inner dimension).
* Depth–width tradeoff tailored for mobile CPUs.
* Distilled from a bigger BERT but with a **mobile-optimized transformer block**.

**Problem it solves**

Standard BERT is not tuned for mobile hardware. MobileBERT aims for **on-device inference** with good latency and accuracy.

**Toy example (architecture intuition)**

Imagine replacing each big, fat BERT layer with:

* A small projection → big inner layer → small projection back.
  This reduces memory bandwidth while keeping computation expressive, much like mobile-optimized CNNs (MobileNet style).

**One-sentence summary**

> MobileBERT redesigns BERT’s architecture for mobile hardware, achieving a good balance of speed, size, and accuracy on edge devices.

---

# Final Timeline Table

| Year  | Model / Family     | Objective / Architecture Change                       | Main Problem Solved                                        |
| ----- | ------------------ | ----------------------------------------------------- | ---------------------------------------------------------- |
| 2018  | BERT               | MLM + NSP on large corpora, bidirectional encoder     | General-purpose language understanding for many tasks      |
| 2018  | mBERT              | Multilingual MLM + NSP across 100+ languages          | Single model for many languages and cross-lingual transfer |
| 2019  | RoBERTa            | No NSP, dynamic masking, more data & training         | Stronger performance via better pretraining recipe         |
| 2019  | DistilBERT         | Knowledge distillation from BERT                      | Smaller, faster BERT with similar performance              |
| 2019  | ALBERT             | Factorized embeddings, cross-layer sharing, SOP       | Reduced parameters and better sentence-level modeling      |
| 2019  | SpanBERT           | Span masking and span-focused objectives              | Better phrase/span representations for QA, coreference     |
| 2019  | SciBERT            | BERT-style training on scientific text                | Improved NLP for scientific papers                         |
| 2019  | BioBERT            | BERT further pretrained on biomedical text            | Better biomedical entity and relation understanding        |
| 2019  | ClinicalBERT       | BERT further pretrained on clinical notes             | Stronger clinical/healthcare NLP                           |
| 2019  | XLM-R              | RoBERTa-style multilingual pretraining on CommonCrawl | Strong multilingual performance, especially low-resource   |
| 2020  | ELECTRA            | Replaced Token Detection (RTD) instead of MLM         | More sample-efficient pretraining                          |
| 2020  | BERTweet           | BERT-style model on large Twitter corpus              | Robust NLP on social-media / tweet text                    |
| 2020  | TinyBERT           | Multi-stage knowledge distillation                    | Highly compressed BERT for low-resource devices            |
| 2020  | MobileBERT         | Mobile-optimized architecture with distillation       | On-device BERT-level NLP with good latency                 |
| 2021+ | DeBERTa (v1/v2/v3) | Disentangled attention, relative positions, RTD (v3)  | Better attention & efficiency; strong SOTA performance     |

---

# ASCII Evolution Diagram (Text Flowchart)

```text
                         +-------------------+
                         |       BERT        |  (2018)
                         |  MLM + NSP, EN    |
                         +---------+---------+
                                   |
           +-----------------------+------------------------+
           |                        |                       |
           |                        |                       |
   +-------v-------+        +-------v-------+       +------v------+
   |   RoBERTa     |        |    mBERT      |       |  Domain     |
   | Better recipe |        | Multilingual  |       | Adaptation  |
   +-------+-------+        +-------+-------+       | (Bio/Sci/   |
           |                        |               | Clinical/   |
           |                        |               | Tweet)      |
   +-------v-------+        +-------v-------+       +------+------+
   |   XLM-R       |        |  (other multi)|              |
   | Stronger multi|        |  not listed   |              |
   +---------------+        +---------------+              |
                                                           
                                                           
           Efficiency / Compression Branch                  |
   +------------------+-------------------------+           |
   |                  |                         |           |
+--v---------+   +----v-----+            +------v-----+    |
| DistilBERT |   | ALBERT   |            | TinyBERT   |    |
| Distilled  |   | Shared   |            | + MobileBERT    |
+------------+   | params   |            | (on-device) |   |
                 +----------+            +------------+    |
                                                           
                                                           
           Structure / Objective Innovations                |
   +---------------------+--------------------------+       |
   |                     |                          |       |
+--v--------+       +----v----+               +-----v----+ |
| SpanBERT  |       | ELECTRA |               | DeBERTa  |-+
| Span mask |       | RTD     |               | Disentang|
+-----------+       +---------+               +----------+
```

* **BERT** is the root.
* **RoBERTa, ELECTRA, DeBERTa**: focus on **training objectives & attention**.
* **DistilBERT, ALBERT, TinyBERT, MobileBERT**: focus on **efficiency & compression**.
* **SpanBERT, DeBERTa**: focus on **richer structure (spans, positions)**.
* **mBERT, XLM-R**: focus on **multilinguality**.
* **BioBERT, ClinicalBERT, SciBERT, BERTweet**: focus on **domain adaptation**.



---

# **BERT Variants — Objective & Innovation Comparison Table**

| Model            | Year | Pretraining Objective                                | Key Architectural Change                       | Main Innovation / Purpose                            |
| ---------------- | ---- | ---------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| **BERT**         | 2018 | **MLM + NSP**                                        | Standard Transformer encoder                   | Baseline bidirectional language understanding        |
| **RoBERTa**      | 2019 | **MLM only (no NSP), dynamic masking**               | Same as BERT                                   | Stronger pretraining recipe: more data, no NSP       |
| **DistilBERT**   | 2019 | **Distillation loss + MLM**                          | 6-layer distilled version                      | Smaller, faster BERT via knowledge distillation      |
| **ALBERT**       | 2019 | **MLM + Sentence Order Prediction (SOP)**            | Embedding factorization + cross-layer sharing  | Huge parameter reduction without loss of performance |
| **SpanBERT**     | 2019 | **Span-Masking MLM + Span-Boundary Objective**       | Span-aware inputs                              | Better span/phrase representations for QA, NER       |
| **ELECTRA**      | 2020 | **Replaced Token Detection (RTD)**                   | Generator + Discriminator                      | Far more sample-efficient training than MLM          |
| **DeBERTa v1**   | 2021 | **Enhanced MLM**                                     | Disentangled attention + enhanced mask decoder | Separates content & position for richer attention    |
| **DeBERTa v2**   | 2021 | **Enhanced MLM**                                     | Optimized embeddings & training                | Stronger representation + more stable training       |
| **DeBERTa v3**   | 2021 | **RTD (ELECTRA-like) + MLM**                         | Same as v2 + RTD                               | Best performance: combines DeBERTa arch + RTD        |
| **BioBERT**      | 2019 | **MLM + NSP (domain pretraining)**                   | Same as BERT                                   | Biomedical language adaptation                       |
| **ClinicalBERT** | 2019 | **MLM + NSP (domain pretraining)**                   | Same as BERT                                   | Clinical notes adaptation                            |
| **SciBERT**      | 2019 | **MLM + NSP (new vocab)**                            | Scientific vocabulary                          | Scientific article adaptation                        |
| **BERTweet**     | 2020 | **MLM + NSP (tweet pretraining)**                    | Tweet tokenizer                                | Social media/Twitter text adaptation                 |
| **mBERT**        | 2018 | **MLM + NSP (multilingual)**                         | Shared vocab across 104 languages              | Multilingual cross-lingual transfer                  |
| **XLM-R**        | 2019 | **MLM only (RoBERTa multilingual)**                  | RoBERTa-style training                         | High-performance multilingual model                  |
| **TinyBERT**     | 2020 | **Two-stage distillation (MLM + task distillation)** | Compressed architecture                        | Extremely small & efficient for edge devices         |
| **MobileBERT**   | 2020 | **Distillation + MLM**                               | Inverted-bottleneck Transformer                | Mobile-optimized on-device BERT                      |



