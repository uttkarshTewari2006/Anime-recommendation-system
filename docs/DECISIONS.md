# Decisions

## 2026-10-02 - Item 1 Evaluation Protocol

**Status:** Approved, implemented, and evaluated. The user selected Option 1.

**Selected protocol:** For each eligible user, hold out one anime rated at least 7 as the target. From the remaining training history, sample up to two distinct anime rated at least 7 as independent seeds. Evaluate each seed-target pair separately, then average pair-level Recall@10 and NDCG@10 within each user before aggregating across users. Exclude every title in the user's training history from candidate lists for both item-CF and popularity.

**Why:** This directly measures whether a positively rated seed retrieves another positively rated title, unlike the previous unrelated first-row seed / held-out-target setup. It preserves a comparable popularity baseline and prevents users with more seed pairs from dominating the overall metrics.

**Reproducibility:** Interaction holdout seed 42; positive seed sampling seed 2026; paired user bootstrap seed 31415 with 2,000 resamples. Exact eligibility counts and run outputs will be recorded in `RESULTS.md` after execution.

**Alternatives considered:**
- Score candidates from the user's full positive history. More representative of personalized recommendation, but it tests profile aggregation rather than item-to-item similarity.
- Hold out several positive targets per user. Uses more targets but requires more complex split accounting and may overweight prolific users.

**Decision scope:** Item 1 only. Do not start catalog-size or model-change experiments until separately approved.

**Execution result:** Item cosine Recall@10 0.157367 (95% CI 0.155249-0.159561) and NDCG@10 0.093584 (0.092068-0.095070); popularity Recall@10 0.139692 (0.136961-0.142408) and NDCG@10 0.075972 (0.074333-0.077629). Full configuration is in `RESULTS.md`.
