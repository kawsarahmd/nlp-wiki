# Understanding Greedy Search and Beam Search in Language Models

When a language model writes a sentence, it does not know the full answer in advance. Instead, it predicts the next word step by step. At each step, the model gives a list of possible next words along with their probabilities. How the model chooses the next word is called the decoding strategy. Two common decoding methods are greedy search and beam search. Both follow simple logic, but they behave differently and produce different quality of text.

## Greedy Search: The Shortest Path to the Next Word

Greedy search is the simplest strategy. At every step, the model looks at all possible next words and picks the one with the highest probability. It does not think about the future or how the sentence may evolve later. It only cares about choosing the best word right now.

For example, imagine the model is completing a sentence that starts with “The cat”. At the next step, the model may assign probabilities like:

* “is” → 0.40
* “was” → 0.30
* “the” → 0.20
* “sat” → 0.10

Greedy search will choose “is”, because it is the most likely next word. This looks reasonable, but greedy search can get stuck later. For instance, after “The cat is”, the model may again think that “the” is the highest-probability option. So it chooses:

* “The cat is the”

This can go on and create repetitions like:

* “The cat is the cat is the cat…”

Greedy search is fast and simple, but it can produce low-quality or repetitive sentences because it makes decisions moment by moment without considering the bigger picture.

## Beam Search: Looking Ahead to Choose Better Sentences

Beam search tries to fix greedy search’s short-sightedness. Instead of keeping only one best sentence, beam search keeps several possible sentences at the same time. This number is called the beam width. At each step, every active sentence is expanded with the most likely next words. Then, only the top sequences with the highest total probability are kept.

Let’s use a simple beam width of 3 for demonstration.

### Step 1: Predict first words

Assume the model gives:

* “The” → 0.60
* “A” → 0.30
* “This” → 0.10

Beam search keeps all three because the beam width is 3.

### Step 2: Expand each sentence

Now expand each one:

“The” can become:

* “The cat” → 0.60 × 0.40 = 0.24
* “The dog” → 0.60 × 0.30 = 0.18

“A” can become:

* “A man” → 0.30 × 0.50 = 0.15
* “A boy” → 0.30 × 0.30 = 0.09

“This” can become:

* “This is” → 0.10 × 0.80 = 0.08
* “This boy” → 0.10 × 0.10 = 0.01

Now we have six total expansions. Beam search picks the top three based on probability:

1. The cat → 0.24
2. The dog → 0.18
3. A man → 0.15

These three become the new active beam. The rest are dropped.

At each step, this process continues: expand, score, and keep only the best few. By the end, beam search usually finds a more meaningful, complete, and natural sentence than greedy search. It avoids the short-sighted decisions that greedy search often makes.

## A Simple Comparison

Suppose the model is trying to complete the sentence:
“The climate crisis”

Greedy search might continue like this:

“The climate crisis is a major issue and it is a major issue and it is a major issue…”

It tends to repeat because it always picks the immediate top word.

Beam search might produce:

“The climate crisis is a growing challenge caused by rising temperatures and human activities.”

This is longer, clearer, and more meaningful because beam search looks at several possible paths, not just one.

## When to Use Each Method

Greedy search is useful when speed is the most important factor. It is extremely fast and works fine for short or factual outputs.

Beam search is better when the quality of the final sentence matters more than speed. It is often used in tasks such as translation, summarization, long text generation, and question answering where a complete, coherent output is important.


## Why Large Language Models Use Log-Probabilities

When a large language model generates text, it predicts the next word one token at a time. At each step, the model produces a set of probabilities showing how likely each possible next token is. These probabilities are often very small numbers such as 0.1, 0.05, 0.01, or even smaller. This becomes a problem because a full sentence may contain 50, 100, or even more tokens. If we try to calculate the probability of an entire sentence, we must multiply all these small numbers together. The result becomes so tiny that computers cannot accurately represent it. This is known as floating-point underflow.

To understand the problem, imagine a short sentence of ten tokens where each token has a probability of around 0.05. If we multiply 0.05 ten times, the number becomes extremely small, so small that it rounds down to zero inside most computer systems. Modern language models generate much longer sequences, which makes the problem even worse. A hundred tokens with probabilities around 0.05 would create a number far beyond what a computer can store. This means that if we stick to normal probabilities, we lose information because the values collapse to zero.

Logarithms solve this problem. Instead of multiplying probabilities, language models convert each probability into its logarithm. The logarithm of a small number is still small, but not dangerously small. For example, while 0.05 raised to the tenth power becomes an almost zero-sized number, the log of 0.05 multiplied by ten becomes a manageable negative value around -30. Computers can handle such numbers easily. This simple transformation keeps the calculation stable and avoids underflow.

Another advantage of logs is that they turn multiplication into addition. Instead of multiplying small probabilities repeatedly, the model adds their log values. Adding a series of numbers like -3, -2, or -1 is much safer and more efficient than multiplying tiny decimal values. This is why during decoding methods like beam search, log-probabilities are used. Beam search explores multiple possible sequences at once. Without log-probabilities, the probability of each sequence would underflow to zero very quickly, making comparison impossible. Using log-probabilities allows beam search to compare and rank different candidate sequences accurately.

This approach also keeps the model’s output stable. When two candidate sequences have to be compared, their log-probabilities remain large enough to distinguish clearly. For example, one sequence might have a total log-probability of -30 and another might have -35. These numbers are easy to compare, while the corresponding original probabilities would be so small that both would look like zero to the computer.

In summary, the reason large language models use log-probabilities is simple. Probabilities in language generation are tiny, and multiplying them repeatedly causes numerical issues. Logarithms solve this by turning multiplication into addition and keeping all values within a safe numeric range. This allows the model to compute long sequences, compare different outputs, and maintain accuracy throughout the decoding process. By using log-probabilities, language models avoid underflow, remain stable, and produce meaningful text efficiently.
