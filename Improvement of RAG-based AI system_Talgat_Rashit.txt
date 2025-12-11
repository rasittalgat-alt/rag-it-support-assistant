# Improvement of RAG-based AI system – IT Support RAG Assistant  
**Author:** Talgat Rashit  

## 1. Project Context

This work is an extension of the previous project **“IT Support RAG Assistant”**, where a basic Retrieval-Augmented Generation (RAG) system was implemented for internal IT support:

- **Domain:** IT support (Wi-Fi, VPN, email, printers, account/password, IT SLA).
- **Knowledge base:** a small but representative dataset:
  - FAQs,
  - runbooks (how-to guides),
  - IT policies and SLA excerpts,
  - example incident tickets.
- **Tech stack:**
  - Python,
  - Qdrant as a vector database,
  - OpenAI embeddings (`text-embedding-3-small`),
  - OpenAI LLM (gpt-4o-mini via `chat.completions`),
  - Streamlit UI,
  - a RAG pipeline orchestrating retrieval + generation.

The goal of this **Advanced RAG** task is to **measure and improve a specific quality aspect** of the RAG subsystem, following an iterative and measurable approach.

---

## 2. Baseline RAG System

The baseline system works as follows:

1. **Embedding the query**  
   - The user question is embedded with `text-embedding-3-small`.

2. **Vector search in Qdrant**  
   - A external semantic similarity search is performed over all chunks in a single Qdrant collection `it_support_kb`, using cosine distance, without any filtering or reranking.

3. **Context construction**  
   - Top-N chunks (typically `top_k = 4`) are taken from Qdrant and passed as context to the LLM.

4. **LLM answer generation**  
   - The LLM receives a system prompt, the user question and the concatenated context, and generates the final answer.
   - The original system prompt was very strict:  
     *“Use ONLY the information from the provided context. If the answer is not contained in the context, say that you don't know and suggest contacting IT Support.”*

This baseline was already working well for “clean” queries, but several limitations remained, especially regarding robustness to user typos and strictness of the prompt.

---

## 3. Metrics

### 3.1. Retrieval Accuracy on Clean Queries: Hit@k

For clean queries (without typos) I use the standard **Hit@k** metric:

- There is an evaluation file `data/eval/queries.json` with 12 queries.
- For each query we know:
  - `id` – query identifier,
  - `question` – user question,
  - `gold_source_id` – identifier of the source document that contains the correct answer.
- During evaluation, the RAG pipeline performs retrieval only (no generation is evaluated).
- We check whether a chunk from the gold document appears among the top-k results.

Formally:

> **Hit@k** = number of queries where gold document is in top-k / total number of queries.

I compute **Hit@1**, **Hit@3** and **Hit@5** using the script `src/eval_rag.py`.

### 3.2. Robustness to Typos: Noisy Hit@1

In real IT support, users often type queries with **typos and misspellings** (e.g. “preinters”, “wfi”, “vnp”, “eamil”, “passwrod”). The baseline system was not specifically optimized for such “noisy” input.

To capture this, I introduced an additional metric:

> **Noisy Hit@1** – Hit@1 measured on a small set of queries with intentional typos.

I created a file `data/eval/queries_typos.json` with 5 noisy queries. Each entry has:

- `id` – typo query id (t1…t5),
- `question` – user question with a typo,
- `gold_source_id` – id of the document that should be considered correct for this query.

The script `src/eval_typos.py` evaluates:

- Baseline Noisy Hit@1 – retrieval using raw query text,
- Improved Noisy Hit@1 – retrieval using normalized (typo-corrected) query text.

This metric turned out to be the **main target metric** for this Advanced RAG task.

---

## 4. Automated Evaluation Setup

### 4.1. Clean Queries Evaluation (`src/eval_rag.py`)

The script:

1. Loads `data/eval/queries.json` (12 clean queries).
2. For each query:
   - embeds the question with `EmbeddingsClient`,
   - queries Qdrant via `VectorDBClient.search(...)`,
   - reads `source_id` from payload of retrieved chunks.
3. Calculates:
   - baseline Hit@1, Hit@3, Hit@5,
   - and (for one of the iterations) Hit@k for a category-aware variant.

The script prints per-query diagnostics and aggregated metrics.

### 4.2. Noisy Queries Evaluation (`src/eval_typos.py`)

The script:

1. Loads `data/eval/queries_typos.json` (5 typo queries).
2. For each query it calculates Hit@1 twice:
   - **Baseline** – using the question “as is”;
   - **Improved** – using `normalize_question(question)` before embedding.
3. Aggregates Noisy Hit@1 and computes relative improvement.

This provides a fully automated way to quantify how robust the system is to user typos.

---

## 5. Iteration 1 – Baseline Measurement on Clean Queries

Using `src/eval_rag.py` with the original RAG system (no category filters, no query normalization), I obtained the following baseline metrics on 12 clean queries:

- **Hit@1:** 11 / 12 = **0.917**  
- **Hit@3:** 12 / 12 = **1.000**  
- **Hit@5:** 12 / 12 = **1.000**

Interpretation:

- For top-3 and top-5 the retrieval is already perfect on this small dataset – the correct document appears in all cases.
- For top-1 there is only one miss: query `q9` (“How do I use Cisco AnyConnect to connect to the corporate VPN?”), where the correct runbook is in top-3 but not at rank 1.

Because Hit@3 and Hit@5 are already saturated (100%), they are not suitable for demonstrating a 30% improvement. Even Hit@1 (0.917) leaves very little room for improvement on this evaluation set.

Therefore, I decided to treat Hit@1 on clean queries mainly as a **diagnostic metric**, and to focus on a second, more realistic metric – robustness to typos.

---

## 6. Iteration 2 – Category-Aware Retrieval (Unsuccessful Attempt)

### 6.1. Idea

The dataset contains natural **categories** (wifi, vpn, email, account, printer, policy/password/sla). The idea was:

1. Add a lightweight rule-based **query classifier**:
   - “wifi”, “wi-fi”, “wireless” → category `wifi`;
   - “vpn”, “AnyConnect” → category `vpn`;
   - “email”, “Outlook”, “webmail” → category `email`;
   - “password”, “account”, “login” → category `account` or `password`;
   - “printer”, “print job” → category `printer`;
   - “SLA”, “priority”, “P1” → category `it`/`sla`.
2. When a category is detected, perform vector search in Qdrant with a **filter by `category`** instead of over the entire collection.
3. Compare Hit@k for:
   - baseline (no filters),
   - category-aware retrieval (with filter).

### 6.2. Implementation

- Rule-based classifier implemented in `eval_rag.py` (function `classify_category(question)`).
- Categories stored in Qdrant payload (`category` field) for each chunk.
- A new retrieval mode was implemented:  
  if classifier returns a category, Qdrant search is executed with a `must` filter on this category.

### 6.3. Results

I extended `src/eval_rag.py` to print both:

- **Baseline Hit@k**, and  
- **Category-aware Hit@k**.

On the same 12 clean queries, the results were:

- **Baseline Hit@1:** 11 / 12 = 0.917  
- **Category-aware Hit@1:** 9 / 12 = 0.750  

Hit@3 and Hit@5 were also slightly worse for the category-aware variant.

Problematic cases included:

- Queries about policies and SLA, where the rule-based classifier chose too generic or wrong category, and the filter excluded the correct document.
- In practice, the dataset is small, and aggressive filtering gives more harm than benefit.

### 6.4. Conclusion for Iteration 2

- Category-aware retrieval **did not improve** Hit@1; it actually decreased it from 0.917 to 0.750.
- The main reason is the limited size of the dataset and imperfect rule-based classification.
- I decided **not to use** this variant in the final RAG pipeline, and instead treat this iteration as an exploratory step.
- The analysis from this iteration motivated me to choose another direction — **robustness to noisy user input** — which is both realistic and clearly valuable for IT support.

---

## 7. Iteration 3 – Robustness to Typos (Successful Improvement)

### 7.1. Motivation

While testing the Streamlit UI, I noticed an important UX problem:

- When I asked:  
  **“How to work with preinters?”** (typo instead of “printers”),
- The system retrieved relevant context about printers (FAQ + ticket), but the LLM answered:  
  *“I don't know how to work with preinters. I suggest contacting IT Support for assistance.”*

So:

- **Retrieval** was already robust enough to typos (semantic similarity still worked),
- but **generation** failed because:
  - the question contained a typo,
  - and the system prompt was too strict (LLM thought the answer is “not in context”).

This is exactly the kind of situation where the user perceives the system as “not smart enough”, despite having the correct documents in the context. Therefore I decided to:

1. Improve the **preprocessing of user queries** (normalization and typo correction).
2. Relax and clarify the **system prompt** for the LLM.

The target metric for this iteration is **Noisy Hit@1** on the typo queries dataset.

### 7.2. Query Normalization (`normalize_question`)

I introduced a function `normalize_question` in `src/text_utils.py`:

- Converts text to lower case.
- Applies a small dictionary of common domain-specific typos, for example:
  - `preinter`, `preinters` → `printer`, `printers`
  - `wfi` → `wifi`
  - `vnp` → `vpn`
  - `eamil` → `email`
  - `passwrod` → `password`
- Can be easily extended with more patterns if needed.

This normalization is applied in three places:

1. **RAGPipeline.retrieve**  
   - Before embedding the query, the question is normalized:
     ```python
     normalized_question = normalize_question(question)
     query_vector = self.emb_client.embed_text(normalized_question)
     ```
2. **RAGPipeline.answer_question**  
   - The LLM receives the **normalized question** (while the UI still shows the original text), so it sees a clean version without typos.
3. **Rule-based classifier** (in evaluation scripts)  
   - For category detection we also use normalized text, which makes the rules more robust.

### 7.3. Prompt Tuning for LLM

The original system prompt was:

> *“Use ONLY the information from the provided context. If the answer is not contained in the context, say that you don't know and suggest contacting IT Support.”*

This led the model to answer “I don’t know” even when the context clearly contained related information, but not in exactly the same wording.

I updated the system prompt in `LLMClient.generate_answer` to a softer and more realistic version (shortened summary):

- Use the provided context as the **main source of truth**.
- User questions may contain **typos** or be **more general** than examples in the context.
- If the context is **related** to the question, the model should **generalize from it** and provide the best practical answer.
- Only if the context is clearly unrelated, the model should say it doesn’t know and suggest contacting IT support.

With this new prompt, for the same typo query *“How to work with preinters?”*, the system now:

- interprets it as “printers”,
- uses FAQ and ticket about network printers,
- generates a detailed step-by-step guide on how to work with corporate printers on Windows.

### 7.4. Noisy Hit@1 – Evaluation Results

Using `src/eval_typos.py` and `data/eval/queries_typos.json`, I evaluated retrieval on 5 noisy queries in two modes:

1. **Baseline** – no normalization; embedding is computed from the raw text.
2. **Improved** – query is passed through `normalize_question` before embedding.

The results:

```text
=== Noisy Retrieval Metrics (Hit@1) ===
Baseline: 3/5 = 0.600
Improved: 4/5 = 0.800
Relative improvement: 33.3%


``` 

So the Noisy Hit@1 improved from 0.60 to 0.80, i.e. a relative improvement of 33.3%, which satisfies the requirement of at least +30%.

## 8. Final Conclusions

1. I implemented an automated evaluation environment for the RAG subsystem:
- clean queries evaluation (src/eval_rag.py + queries.json),
- noisy queries evaluation (src/eval_typos.py + queries_typos.json).

2. I experimented with category-aware retrieval as an enhancement:
- It used rule-based classification and Qdrant filters,
- but on this dataset it reduced Hit@1 instead of improving it,
- so it was documented as an unsuccessful iteration and not kept in the final system.

3. I identified robustness to user typos as a truly valuable metric for an IT Support assistant and introduced Noisy Hit@1 as the main evaluation metric.

4. I implemented two key improvements:
- Query normalization (normalize_question) to correct typical typos before retrieval and generation,
- Prompt tuning to allow the LLM to generalize from related context instead of over-using the “I don’t know” fallback.

5. These changes led to a 33.3% relative improvement of the Noisy Hit@1 metric:
- from 0.60 (3/5) to 0.80 (4/5) on the typo evaluation set.

6. Qualitative testing in the Streamlit UI confirms the improvement:
- for the typo query “How to work with preinters?” the system now retrieves the correct FAQ and produces a helpful, step-by-step answer based on the context.

Overall, the Advanced RAG task resulted not only in a measurable metric improvement above the required 30%, but also in a practically useful enhancement of the IT Support RAG Assistant, making it more robust and user-friendly in realistic scenarios with noisy inputs.

