# How LLMs Are Trained — Complete Study Notes

*A reference guide combining classroom notes with supplementary research from industry blogs (Jay Alammar, MLOps Community, Cameron Wolfe, RLHF practitioners).*

---

## Table of Contents
1. [The Big Picture](#1-the-big-picture)
2. [Super Simple Version (Analogy-Based)](#2-super-simple-version)
3. [Phase 1: Pretraining — Deep Dive](#3-phase-1-pretraining)
4. [The Transformer Architecture (Mechanics Behind Pretraining)](#4-the-transformer-architecture)
5. [Phase 2: Supervised Fine-Tuning (SFT)](#5-phase-2-supervised-fine-tuning-sft)
6. [Phase 3: Alignment (RLHF)](#6-phase-3-alignment-rlhf)
7. [What the Model Actually Learns](#7-what-the-model-actually-learns)
8. [Case Study: LLaMA 1 Training Data](#8-case-study-llama-1-training-data)
9. [Beyond RLHF: DPO, KTO, RLAIF](#9-beyond-rlhf-dpo-kto-rlaif)
10. [Key Terms Glossary](#10-key-terms-glossary)
11. [Quick-Recall Summary Table](#11-quick-recall-summary-table)
12. [Likely Interview / Exam Questions (with Answers)](#12-likely-interview--exam-questions)

---

## 1. The Big Picture

LLMs are trained in **three major phases**:

| # | Phase | Purpose |
|---|---|---|
| 1 | **Pretraining** | Learn language itself — grammar, facts, reasoning patterns |
| 2 | **Supervised Fine-Tuning (SFT)** | Learn to follow instructions and behave like an assistant |
| 3 | **Alignment (RLHF)** | Learn to be helpful, safe, and prefer "better" responses |

Each phase builds on the previous one. You cannot skip pretraining — it's where ~99% of the compute cost and ~99% of the "intelligence" comes from. SFT and RLHF are comparatively cheap "steering" phases.

```
Raw Internet Text
      │
      ▼
 [PRETRAINING]  →  Base / Foundation Model (knows language, not assistant-like)
      │
      ▼
 [SFT]  →  Instruction-Following Model (behaves like an assistant)
      │
      ▼
 [RLHF]  →  Aligned Model (helpful, safe, prefers better answers) = ChatGPT/Claude-like
```

---

## 2. Super Simple Version

Think of training an LLM like **teaching a child to complete sentences**.

### Step 1: Show it massive amounts of text
The model reads (as *patterns*, not like a human):
- Books, websites, articles, code, conversations
- Millions to billions of statements

### Step 2: Play a guessing game
Guess the next word.
```
Input:   "The sky is"
Correct: "blue"

Guess "car"  → Wrong
Guess "blue" → Correct
```
Wrong guesses trigger **backpropagation** (adjusting internal numbers slightly). This happens **trillions of times**.

### Step 3: Adjust the numbers
- Inside the model are billions of numbers: **weights** and **biases**.
- Every wrong guess → numbers change slightly.
- Over time → the model gets very good at predicting the next word.

### Step 4: Teach it to follow instructions
After basic training, the model knows language patterns but doesn't know how to *behave* like an assistant (ChatGPT, Claude). So it's trained on examples like:
```
User: Explain gravity simply
Assistant: Gravity is a force that pulls objects together.
```

### Step 5: Teach it to be safe
Humans compare answers and rank good vs. bad responses. The model learns to:
- Be polite
- Avoid harmful responses
- Say "I don't know" when unsure

---

## 3. Phase 1: Pretraining

**This is the Foundation Phase — where most of the model's intelligence is learned.**

**Objective:** Learn to predict the next token (**next-token prediction** / **Causal Language Modeling**).

**Learning type:** Self-supervised — no manual labels needed. The "label" is simply the next word already present in the text.

### Step-by-step pipeline

**Step 1 — Collect massive data**
Sources:
- Web pages
- Books
- Wikipedia
- Code
- Academic articles

Scale: **trillions of tokens** (hundreds of billions to 10+ trillion depending on the model).

**Step 2 — Tokenization**
`Text → Tokens`
Text is broken into sub-word units (tokens) that the model can process numerically.

**Step 3 — Forward pass through Transformer**
```
Tokens → Embeddings → Transformer layers → Logits (raw scores) for next token
```

**Step 4 — Compute loss**
Compare predicted probability distribution with the actual next token using **cross-entropy loss**.
- Correct word predicted with **low probability** → **high loss**
- Correct word predicted with **high probability** → **low loss**

**Step 5 — Backpropagation**
- Gradients are calculated
- Weights are updated using an optimizer (commonly **AdamW**)
- This repeats **billions of times**

### What happens during pretraining
- **All** weights are updated
- Billions of parameters adjust gradually
- The model learns **statistical patterns in language**

### Extra depth (from research)

- **ULMFiT (2018)** was the first major work to show pretrain-then-finetune works well for NLP — borrowed the idea from ImageNet pretraining in computer vision.
- **InstructGPT (2022)** formalized the 3-stage pipeline (Pretrain → SFT → RLHF) that underlies ChatGPT and most modern LLMs.
- **Chinchilla finding (DeepMind, 2022):** More *data* often matters more than more *parameters*. A 70B-parameter model trained on 1.4T tokens (Chinchilla) beat a 280B-parameter model (Gopher) trained on less data — this reshaped how labs allocate training compute (the "compute-optimal" scaling laws).
- **Data quality is critical:** Bad or unusual data segments can cause "loss spikes" that destabilize training. This is why **deduplication**, **domain balancing**, and **curriculum learning** (ordering data by difficulty) are active research areas.
- **Continual pretraining:** Later models (e.g., LLaMA-2) can resume pretraining from an earlier checkpoint on new data — updating knowledge without training entirely from scratch.

---

## 4. The Transformer Architecture

*(The mechanism operating "under the hood" during the forward pass in pretraining, SFT, and inference.)*

### High-level structure
- **Encoder stack** + **Decoder stack** (6 layers each in the original "Attention Is All You Need" paper)
- Modern LLMs like GPT are **decoder-only** — they don't need a separate encoder

**Each Encoder block:**
```
Self-Attention → Feed-Forward Neural Network (FFN)
```
The FFN is applied independently to each word position — this is what allows parallelization.

**Each Decoder block:**
```
Self-Attention → Encoder-Decoder Attention → Feed-Forward
```

### Self-Attention — step by step
1. Each word's embedding is multiplied by three trained weight matrices → produces a **Query (Q)**, **Key (K)**, and **Value (V)** vector (dimension 64, vs. embedding dimension 512).
2. **Score** = dot product of a word's Query with every word's Key (including its own).
3. Divide scores by √64 = 8 (stabilizes gradients — this is the "scaled" in "scaled dot-product attention").
4. Apply **softmax** to scores → normalized weights that sum to 1.
5. Multiply each Value vector by its softmax weight.
6. Sum all weighted Value vectors → the output for that word's position.

**Why this matters:** This is how a model figures out that "it" in *"The animal didn't cross the street because it was too tired"* refers to "animal," not "street."

### Multi-Head Attention
Instead of one Q/K/V set, use **8 separate heads**, each trained to attend to different types of relationships. Outputs are concatenated and passed through one more weight matrix to condense back into the right shape.

### Positional Encoding
Attention has no inherent sense of word order, so a sine/cosine-based vector is **added** to each embedding to encode position in the sequence.

### Residual Connections + Layer Normalization
Every sub-layer (attention, FFN) has a skip/residual connection around it, followed by layer normalization — critical for stabilizing training in deep networks.

### Final output layer
```
Decoder output → Linear layer (projects to vocab-size logits) → Softmax → probability distribution over every possible next token
```
The highest-probability token is selected (or explored via beam search / sampling strategies like top-k, top-p, temperature).

### Loss & backpropagation (tie-back to Section 3)
Compare predicted distribution vs. actual next-token (one-hot) target → **cross-entropy loss** → backpropagate → update weights.

---

## 5. Phase 2: Supervised Fine-Tuning (SFT)

**Also called: Instruction Tuning**

After pretraining, the model knows language but **not how to behave as an assistant**.

### What SFT teaches
- Following instructions
- Formatting answers properly
- Being conversational

### Step 1 — Create an instruction dataset
Humans write curated examples:
```
User: Explain photosynthesis simply.
Assistant: ...

User: Write a python function to reverse a string.
Assistant: ...
```

### Step 2 — Train like pretraining, but with pairs
- The model is shown `Input Instruction + Expected Output`
- It learns to reproduce that pattern
- **Important:** SFT uses the **exact same training objective as pretraining** — next-token prediction. The only difference is the *data* (curated instruction pairs vs. raw web text). In practice, loss is often applied only to the *response* portion, not the prompt.

### Extra depth (from research)
- **Cost:** SFT is roughly **100x cheaper** than pretraining (thousands of dollars vs. hundreds of thousands+).
- **LIMA paper finding:** Only **~1,000 carefully curated examples** were enough to produce a highly competitive model — proving that **data quality/diversity matters far more than raw quantity** for SFT.
- **LLaMA-2** used ~27,540 manually curated dialogue examples for SFT. Interestingly, once SFT performance plateaued, the model's *own* generations became good enough to be used as further SFT data — so further effort shifted toward collecting **preference data for RLHF** instead.
- **Imitation models** (Alpaca, Vicuna, Koala): a cheap approach where you take an open base model and fine-tune it on outputs collected from a stronger proprietary model (like GPT-4). Effective but was shown to underperform true RLHF-aligned models on deeper reasoning tasks.
- **SFT alone is not enough** — LLaMA-2 results showed that even after strong SFT, adding RLHF produced a further large jump in helpfulness and safety scores.

---

## 6. Phase 3: Alignment (RLHF)

**Goal:** Make the model helpful **and** safe.

Even after Pretraining (knows language) + SFT (follows instructions), problems remain:
- May give unsafe answers
- May be too verbose
- May be rude

**Core idea of RLHF:** Instead of telling the model the *exact correct answer*, we tell it *which answer is better*. This works for qualities that are hard to define with rules (e.g., "funnier," "safer," "more helpful") but easy for a human to judge by comparison.

### Step-by-step pipeline

**Step 1 — Generate multiple responses**
For one prompt, the model generates several responses (A, B, C).

**Step 2 — Humans rank the responses**
Example ranking: `A > C > B`

**Step 3 — Train a Reward Model (RM)**
A separate, smaller model is trained to:
- Take `(prompt, response)` as input
- Output a **scalar score**
- Learn from data like "humans prefer A more than B"

This RM acts as an automated stand-in "judge" so a human doesn't need to be in the loop for every training step.

**Step 4 — Reinforcement Learning (usually PPO — Proximal Policy Optimization)**
```
Main LLM generates an answer
        │
        ▼
Reward model scores it
        │
        ▼
High score → reinforce that behavior
Low score  → penalize that behavior
        │
        ▼
Model gradually shifts toward higher-reward responses
```

**Step 5 — KL-Divergence Penalty (critical stabilizer)**
A regularization term that keeps the updated model's outputs from drifting too far from the original SFT model. Without it, the model can "reward hack" — producing gibberish or repetitive output that scores artificially high on the reward model while becoming useless in practice.

### Known failure modes of RLHF
- **Reward hacking** — model exploits weaknesses in the reward model rather than genuinely improving
- **Over-optimization** — pushing too hard against the reward model degrades real quality
- **Loss of output diversity** — model converges to "safe" repetitive answers
- **Mis-specified/misaligned preferences** — if human raters have inconsistent or biased preferences, the model learns those biases too

---

## 7. What the Model Actually Learns

Across all three phases, the model learns:
- Grammar & syntax
- Semantic relationships
- Reasoning patterns
- Coding patterns
- Statistical world knowledge

**Important distinction to remember:** The model does **NOT** store:
- A database of facts
- Symbolic rules

Everything is encoded as **statistical patterns distributed across billions of weights** — not as retrievable, structured facts (this is why LLMs "hallucinate" — they generate the statistically most-likely-sounding text, not looked-up facts).

---

## 8. Case Study: LLaMA 1 Training Data

**Datasets used:**
- CommonCrawl
- C4 (Colossal Clean Crawled Corpus)
- GitHub
- Wikipedia
- Books
- ArXiv
- StackExchange

**Token distribution (Meta's reported percentages):**

| Source | % of Total Tokens |
|---|---|
| Web data (CommonCrawl + C4) | ~67% |
| Books | ~15% |
| Code (GitHub) | ~4.5% |
| Wikipedia | ~4.5% |
| ArXiv | ~2.5% |
| StackExchange | ~2% |

**Key takeaway:** The vast majority of pretraining data (67%) is general web crawl content, with books as the second-largest single source (15%). Code, Wikipedia, ArXiv, and StackExchange together make up only ~13.5% of tokens but contribute disproportionately to the model's reasoning, factual accuracy, and coding ability — this is a good example of why **data composition, not just data volume**, shapes model capability.

---

## 9. Beyond RLHF: DPO, KTO, RLAIF

*(Not in the original classroom notes, but important modern context — these are the techniques that have largely supplemented or replaced classic PPO-based RLHF in many labs.)*

| Method | Key Idea | Advantage over classic RLHF |
|---|---|---|
| **DPO** (Direct Preference Optimization) | Skips training a separate reward model *and* the RL loop entirely — directly optimizes the LLM on preference pairs using a closed-form loss | Simpler pipeline, more stable, cheaper to run |
| **KTO** (Kahneman-Tversky Optimization) | Only needs a binary "good/bad" label per response — no paired A-vs-B comparisons needed | Much easier/cheaper data collection |
| **RLAIF** (RL from AI Feedback) | Replaces human preference labelers with a stronger AI model as the judge | Scales preference data collection without human annotation cost |

---

## 10. Key Terms Glossary

| Term | Definition |
|---|---|
| **Token** | The basic unit of text a model processes (a word, sub-word, or character depending on the tokenizer) |
| **Embedding** | A numeric vector representation of a token that captures meaning |
| **Logits** | Raw, unnormalized output scores from the model before softmax is applied |
| **Softmax** | A function that converts raw scores into a probability distribution summing to 1 |
| **Cross-Entropy Loss** | The standard loss function for next-token prediction; measures how "wrong" a predicted probability distribution is vs. the true target |
| **Backpropagation** | The algorithm that computes gradients and propagates the error backward through the network to update weights |
| **Weights & Biases** | The billions of learnable numeric parameters inside a neural network |
| **AdamW** | A commonly used optimizer for updating weights during training, incorporating momentum and weight decay |
| **Self-Attention** | The mechanism allowing each token to "look at" and weigh the relevance of every other token in the sequence |
| **Query, Key, Value (Q/K/V)** | The three vectors derived from each token's embedding, used to compute attention scores |
| **Multi-Head Attention** | Running several attention mechanisms ("heads") in parallel, each potentially learning different relationship types |
| **Positional Encoding** | Vectors added to embeddings to give the model a sense of word order |
| **Reward Model (RM)** | A model trained to predict a scalar score reflecting human preference for a given response |
| **PPO** (Proximal Policy Optimization) | The reinforcement learning algorithm most commonly used to fine-tune LLMs against a reward model, using clipped updates for stability |
| **KL Divergence** | A measure of how different two probability distributions are; used in RLHF as a penalty to prevent the model from drifting too far from its SFT starting point |
| **Base/Foundation Model** | The output of pretraining alone — knows language but isn't instruction-tuned or aligned |
| **Chinchilla Scaling Laws** | Research finding that model performance is optimized by balancing parameter count and training data volume, not maximizing parameters alone |

---

## 11. Quick-Recall Summary Table

| Stage | Objective | Data Used | Relative Cost | Output |
|---|---|---|---|---|
| **Pretraining** | Next-token prediction (self-supervised) | Trillions of tokens: web, books, code, Wikipedia, etc. | Very High (hundreds of thousands to millions of $) | Base/Foundation model |
| **SFT** | Next-token prediction (supervised, on curated pairs) | Thousands of curated (instruction, response) examples | Low (~100x cheaper than pretraining) | Instruction-following model |
| **RLHF** | Reward maximization via PPO + KL penalty | Human preference rankings on model output pairs | Low–Moderate (human annotation cost) | Aligned, helpful, safe model |

---

## 12. Likely Interview / Exam Questions

**Q1: What are the three main phases of LLM training?**
> Pretraining, Supervised Fine-Tuning (SFT), and Alignment (RLHF).

**Q2: What is the training objective during pretraining?**
> Next-token prediction — given previous tokens, predict the next one. This is self-supervised since the "label" is just the next word already in the text.

**Q3: What loss function is used, and why?**
> Cross-entropy loss. It compares the model's predicted probability distribution over the vocabulary against the actual next token — low probability on the correct token yields high loss, penalizing the model appropriately.

**Q4: How is SFT different from pretraining if both use next-token prediction?**
> They use the same underlying objective, but different data: pretraining uses massive raw, unlabeled text; SFT uses a smaller, curated set of high-quality (instruction, response) pairs. SFT is also far cheaper computationally.

**Q5: Why can't we skip SFT and go straight to RLHF?**
> SFT gives the model a baseline ability to follow instructions via imitation learning. RLHF then refines *which* of several plausible, already-reasonable answers is better — a distinction that's hard to teach via direct examples alone.

**Q6: What is a reward model and why is it needed?**
> A separate model trained to predict a scalar score reflecting human preference for a given (prompt, response) pair. It's needed because having a human rate every single output during RL training would be far too slow/expensive — the reward model acts as an automated proxy judge.

**Q7: What role does the KL-divergence penalty play in RLHF?**
> It penalizes the model for drifting too far from its SFT starting point during RL fine-tuning, preventing "reward hacking" where the model produces degenerate/gibberish outputs that happen to score highly on the reward model.

**Q8: What does an LLM actually "store" — facts or patterns?**
> Statistical patterns distributed across billions of weights, not a retrievable database of facts or symbolic rules — this is a key reason LLMs can hallucinate confidently.

**Q9: What does the LLaMA 1 data mix reveal about pretraining data composition?**
> That the majority of tokens (67%) come from general web crawl data, with books next largest (15%). Specialized sources like code, Wikipedia, ArXiv, and StackExchange are a small percentage of total tokens but are believed to contribute disproportionately to reasoning and factual capability.

**Q10: What are some newer alternatives to PPO-based RLHF?**
> DPO (Direct Preference Optimization) — skips the reward model and RL loop entirely; KTO — uses simple binary good/bad labels instead of paired comparisons; RLAIF — uses AI judges instead of human labelers.

**Q11: Explain self-attention in one sentence.**
> Each token computes a Query vector and compares it (via dot product) against every other token's Key vector to determine how much attention/weight to give that token's Value vector when building its own contextual representation.

**Q12: Why do we scale attention scores by √d_k (e.g., √64=8)?**
> To stabilize gradients — without scaling, dot products can grow large in magnitude, pushing the softmax into regions with extremely small gradients, which slows/destabilizes training.

---

*End of notes. Recommended review order: Section 1 → 2 → 3 → 4 → 5 → 6 → 7-8 (context) → 12 (self-test).*
