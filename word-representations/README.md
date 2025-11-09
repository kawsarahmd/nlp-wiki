
# The Evolution of Word Representations in Natural Language Processing

Over the last decade, Natural Language Processing (NLP) has evolved from simple word representations to powerful models that can understand meaning, context, and relationships across entire texts. This transformation began with **Word2Vec** and gradually moved through **GloVe**, **fastText**, **ELMo**, **ULMFiT**, and eventually led to **BERT** and **GPT**. Each model contributed an important step in helping machines understand human language more accurately.

---

## Word2Vec: The Beginning of Modern NLP (2013)

---

Before Word2Vec, computers could only see words as separate symbols — they had no idea that some words are related in meaning. Word2Vec changed that by turning words into **numbers** (called vectors) so that words with similar meanings are placed close to each other in a kind of “word space.”

**Main idea:**
Words that appear in similar situations usually have similar meanings.

**Example:**
Take these two sentences:

* The cat sleeps on the sofa.
* The dog rests on the bed.

Word2Vec notices that *cat* and *dog* often appear in similar places in sentences, so it learns that their meanings are related. The same happens with *sofa* and *bed*, which also appear in similar contexts. This helps computers understand which words are alike in meaning, even without knowing what they truly mean.

**Limitation:**
Word2Vec gives the **same vector** to a word no matter how it’s used. For example, the word *Apple* will have one single meaning in the model, whether it appears in “Apple launched a new iPhone” or “I ate a red apple.” Even though one refers to a company and the other to a fruit, Word2Vec cannot tell the difference because it doesn’t look at the context deeply enough.


---

## GloVe: Combining Global and Local Context (2014)

The **GloVe (Global Vectors)** model improved Word2Vec by combining both **local** context (words that appear nearby) and **global** statistics (how frequently words co-occur in the whole text corpus).

**Example:**
If “ice” often appears with words like “cold,” “snow,” and “frozen,” while “steam” often appears with “hot,” “boil,” and “vapor,” then GloVe captures the relationship:

`ice – steam ≈ cold – hot`

This shows that the model not only understands which words are related but also how they relate.

**Limitation:**
Like Word2Vec, GloVe still produces a single fixed vector per word and cannot differentiate between meanings depending on context.

---

## fastText: Understanding Subwords (2016)

Facebook introduced **fastText** to address another problem—handling rare or misspelled words. Unlike Word2Vec, fastText breaks each word into **subword units** (smaller chunks such as character n-grams).

**Example:**
The word *happiness* might be split into small parts such as *hap*, *happ*, *pine*, and *ness*. Even if the model has never seen the word *happiness* before, it can understand it based on these smaller parts.

This makes fastText especially effective for morphologically rich languages and spelling variations.

**Limitation:**
Although it handles unseen words better, it still gives the same vector for a word regardless of its sentence meaning.

---

## ELMo: The Start of Contextual Embeddings (2018)

**ELMo (Embeddings from Language Models)** marked a major shift by introducing **contextual word embeddings**. This means that the same word can have different representations depending on the sentence it appears in.

**Example:**

* He went to the river bank.
* She deposited money in the bank.

Earlier models like Word2Vec would assign the same vector to *bank* in both sentences. ELMo, however, produces two different vectors based on the surrounding words. This was achieved by using **bidirectional LSTMs**, allowing the model to read text both forward and backward to understand full context.

**Strength:** Captures word meaning based on surrounding words.
**Weakness:** Still based on LSTM, which is slower and struggles with long sequences.

---

## ULMFiT: Transfer Learning Comes to NLP (2018)

**ULMFiT (Universal Language Model Fine-tuning)** introduced the idea of **transfer learning** to NLP, which had already revolutionized computer vision. Instead of training a new model from scratch for each task, ULMFiT proposed pretraining a general language model on a large dataset and then fine-tuning it for a specific task, such as sentiment analysis or text classification.

**Example:**
A model is first trained on Wikipedia to learn general English. Then it is fine-tuned on movie reviews to predict whether a review is positive or negative.

This approach significantly reduced the amount of labeled data needed and became the foundation for later transformer models like BERT and GPT.

---

## BERT: Deep Bidirectional Understanding (2018)

**BERT (Bidirectional Encoder Representations from Transformers)** marked the beginning of the **transformer era**. Unlike previous models, BERT reads the entire sentence at once—both left and right contexts—using the **Transformer architecture**.

BERT uses a technique called **Masked Language Modeling (MLM)**. During training, it randomly hides some words in a sentence and asks the model to predict them using the surrounding words.

**Example:**
Input: *The [MASK] barked loudly.*
BERT predicts: *dog*

Because it looks at both sides of the masked word, BERT understands context deeply. It became a foundation for many downstream NLP tasks like question answering, named entity recognition, and summarization.

**Strength:** Strong bidirectional understanding and context awareness.
**Limitation:** Computationally expensive and cannot generate text naturally (it’s an encoder-only model).

---

## GPT: The Rise of Generative Models (2018–Present)

While BERT focuses on understanding text, **GPT (Generative Pretrained Transformer)** focuses on **generating** text. GPT is trained using a **causal language modeling** objective—it learns to predict the next word in a sequence.

**Example:**
Input: *The cat sat on the*
GPT predicts: *mat.*

This simple next-word prediction allows GPT models to generate coherent and fluent paragraphs, complete stories, and even entire conversations. Each new version of GPT—GPT-2, GPT-3, GPT-4, and beyond—has expanded in scale and capability, demonstrating that with enough data and parameters, language models can perform tasks ranging from translation to code generation and reasoning.

**Strength:** Excellent for text generation and creative tasks.
**Limitation:** Lacks full bidirectional context (unlike BERT), though fine-tuning and large-scale training help overcome this.

---

## Summary of the Evolution

| Model    | Year  | Key Innovation                     | Context-Aware | Example                     |
| -------- | ----- | ---------------------------------- | ------------- | --------------------------- |
| Word2Vec | 2013  | Similar words have similar vectors | No            | cat ↔ dog                   |
| GloVe    | 2014  | Combines local and global context  | No            | ice : steam :: cold : hot   |
| fastText | 2016  | Uses subword units                 | No            | understands "happyness"     |
| ELMo     | 2018  | Contextual embeddings (bi-LSTM)    | Yes           | bank (river) ≠ bank (money) |
| ULMFiT   | 2018  | Transfer learning for NLP          | Yes           | fine-tuned for sentiment    |
| BERT     | 2018  | Deep bidirectional transformer     | Yes           | predicts masked word        |
| GPT      | 2018– | Autoregressive text generation     | Yes           | completes sentences         |

---

## Conclusion

The journey from Word2Vec to GPT shows how language models have evolved from simple statistical word relationships to deep contextual understanding and generation. Early models like Word2Vec and GloVe captured basic meaning but lacked context. Later models like ELMo and ULMFiT brought contextual and transferable understanding, while transformer-based models like BERT and GPT revolutionized the field by enabling truly human-like comprehension and generation of text.

Today’s large language models are built upon these foundations, demonstrating how each step—no matter how simple—was essential in shaping the modern era of natural language processing.

converted into a “Related Work” section for your NLP paper.

