# LLM-Assisted Hyperlink Reranking (Algorithm 1)

LLM reranking stage for the Wikipedia hyperlink recommendation pipeline.
Implements **Algorithm 1** from the paper: NGCF top-N candidates → LLM
batch scoring → score fusion → top-K recommendations.

## Pipeline

```
NGCF top-100 candidates
        │
        ▼
  Candidate cards  (qid, label, NGCF score, cross-lingual support)
        │
        ▼
  Batch split (size b = 20)
        │
        ▼
  LLM scoring  (Qwen3 — relevance score 1-5 + reason, per batch)
        │
        ▼
  Aggregate scores across batches
        │
        ▼
  Min-max normalise NGCF scores  +  Min-max normalise LLM scores
        │
        ▼
  Fused score = α · NGCF_norm + (1-α) · LLM_norm
        │
        ▼
  Sort → return top-K
```

## Files

- **`hlink_qwen3_colab.ipynb`** — self-contained Google Colab notebook.
  Loads Qwen3-8B directly on the Colab GPU (no external server required),
  runs the full Algorithm 1 pipeline, and evaluates against NGCF baseline
  (Hit Rate@K, Recall@K, NDCG@K) with an α-fusion ablation.
- **`algo1_vi_rank10_top100_b20_mock-paper-format.json`** — example output
  in the paper's expected JSON ranking format, generated with the offline
  mock client (no LLM calls) for pipeline verification.

## Prompt design

**System prompt:**
> You are an experienced Wikipedia editor. Evaluate whether each candidate
> hyperlink should be inserted into the target article. Consider semantic
> relevance, topical consistency, and Wikipedia linking conventions. Ignore
> popularity unless supported by the article context.

**Expected output (structured JSON):**
```json
{"ranking": [{"qid": "Q123", "score": 5, "reason": "Closely related to the article."}]}
```

## Running on Google Colab

1. Upload `hlink_qwen3_colab.ipynb` to Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run cells in order — cell 3 prompts for `app_recommendations.json` upload.
4. Cell 5 loads Qwen3-8B in 4-bit (fits in T4's 15 GB VRAM).
5. Cell 11 runs a single-article demo; cell 12 runs full evaluation with
   an α ∈ {0.3, 0.5, 0.7} ablation.

**Estimated runtime (T4, Qwen3-8B 4-bit):** ~3–5 min/article.
Use `MAX_ARTICLES` to limit the evaluation set for a quicker run.

## Model

- **Qwen3-8B** (`Qwen/Qwen3-8B`), 4-bit quantised via BitsAndBytes for T4 GPU.
- Thinking mode disabled (`enable_thinking=False`) for clean JSON output.
- Disk-cached responses — interrupted runs can resume without re-querying
  identical prompts.

## Evaluation

Ground truth: held-out candidate at NGCF rank 10 per article.
Metrics: Hit Rate@K, Recall@K, NDCG@K for K ∈ {1, 5, 10}, compared against
the NGCF-only baseline.
