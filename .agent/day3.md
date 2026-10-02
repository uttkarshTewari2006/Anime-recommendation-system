# Day 3 — Advanced Anime Recommender and Project Delivery

## Objective
Upgrade the Day 2 baseline into a more polished recommender system deliverable, document the performance tradeoffs, and package the project so it is easy to run, evaluate, and explain.

---

## Deliverables
- Improved anime recommendation pipeline
- Comparison between baseline and upgraded version
- Final notebook or script-based project
- README-style project summary with setup and usage instructions
- Optional lightweight app or demo if time permits

---

## Recommended Default Plan
- Keep the baseline as item-based collaborative filtering
- Add a better ranking layer or metadata-aware rerank if feasible
- package the model and evaluation outputs into a reproducible project structure
- document the final results and next steps

---

## Step-by-Step Tasks

| Task ID | Task | Type | Agent autonomy | Done when | Human instruction required |
|---|---|---|---|---|---|
| D3-01 | DONE - Review Day 2 baseline performance and identify weak points | Agent-only | Baseline limitations and next steps recorded | `results/day3_summary.md` describes limits and comparison | None |
| D3-02 | DONE - Add genre-aware reranking and positive-history scoring | Hybrid | Genre weight selected with validation metrics | Genre metadata covers all 750 evaluated anime; default genre weight is 0.10 | Agent chose genre metadata and measured it on held-out users |
| D3-03 | DONE - Compare baseline and upgrades using Recall@K and NDCG@K | Agent-only | Same final test users used for every model | `results/day3_metrics.csv` and validation sweep record the comparison | None |
| D3-04 | DONE - Refine the recommended anime function for top-K output | Agent-only | Sorted results, input validation, and seen-item exclusion covered by tests | Exact-title and user-history recommendation paths are implemented | None |
| D3-05 | DONE - Package a reusable script and notebook entry point | Agent-only | One script is the reproducible source of truth | `run_evaluation.py` and `notebooks/anime_item_cf.ipynb` run the evaluation | None |
| D3-06 | DONE - Write project summary and setup instructions | Agent-only | README contains setup, commands, results, and limitations | README and `results/day3_summary.md` updated | None |
| D3-07 | OPTIONAL - Standalone app not added | Hybrid | Notebook launcher and script provide the demo/run workflow | Optional app deferred; no separate app is needed for the requested deliverable | Notebook/script selected as the lightweight interface |
| D3-08 | DONE - Final review and handoff | Agent-only | Tests pass; comparison and next steps documented | Final result summary is recorded below | None |

---

## Detailed Execution Notes

### Task D3-02: Add ranking improvements
This is the main upgrade step. Since the project is still in the 3-day scope, the most practical options are:
- add a metadata-aware rerank using genres or synopsis similarity,
- apply score normalization or filtering,
- or create a simple weighted blend of collaborative score and metadata score.

Decision: use genres for reranking. Genre metadata is populated for all 750 items in the evaluated catalog; the weight was selected on validation and evaluated on a separate test split.

### Task D3-07: Optional demo
This is not required for the MVP, but it helps make the project feel complete.

Decision: keep the workflow script- and notebook-based. A standalone app is optional and was not needed to complete this deliverable.

---

## Success Criteria for Day 3
The day is complete when all of the following are true:
- The baseline is evaluated and compared against an improved version.
- The recommendation function works cleanly and consistently.
- The project structure is organized and reusable.
- Setup and usage notes are available.
- The final output is understandable to a reviewer or future user.

---

## Exit Criteria for the 3-Day Project
The project is considered complete when:
- The dataset has been processed and cleaned properly.
- The baseline item-based recommender is running.
- Similar anime are returned for a selected title.
- Metrics are calculated and documented.
- The code and documentation are organized and readable.
- The user has a clear path to continue or improve the system later.

---

## Recommended Final Output
The best final output for this project is:
- a notebook with the full workflow,
- a reusable Python script or module for the recommendation engine,
- a markdown summary or README with evaluation results,
- optional local demo if the user explicitly wants one.

This keeps the project credible, understandable, and robust without expanding beyond the 3-day scope.

---

## Human Intervention Checklist
- Metadata choice resolved: genres; validation-selected weight 0.10.
- Demo choice resolved: notebook launcher and script; no standalone app.
- Performance priority followed: tuning uses validation data and final metrics use a disjoint test split.

## Completed Day 3 Results

Evaluation uses a 750-anime catalog, one held-out rating of at least 7 per eligible user, and 62,743 users shared across validation and test. The genre weight was selected on validation.

| Model | Genre weight | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Original single-seed cosine | 0.00 | 0.138517 | 0.078878 |
| Positive-history cosine | 0.00 | 0.262738 | 0.158557 |
| Genre hybrid | 0.10 | 0.269863 | 0.161320 |
| Popularity | 0.00 | 0.136701 | 0.073877 |

The genre hybrid gained 0.007124 Recall@10 and 0.002764 NDCG@10 over positive-history cosine. Positive-history aggregation delivered most of the lift over the original single-seed baseline. Run `python run_evaluation.py` to regenerate validation metrics, final metrics, per-user rankings, and the summary under `results/day3_*`.

## Remaining Tasks

- **3b - Neighbor reliability:** Compare co-rater similarity shrinkage against a popularity penalty on the fixed 5,000-title catalog, split, and protocol. Confirm the method before running; log metrics and top-10 neighbors for the fixed seed-title list.
- **4 - Teacher text/data:** Present text-source and metadata-only teacher options with tradeoffs, recommend one, and wait for approval before downloading data or building the cross-encoder. List any required manual setup.
- **5 - Schedule:** Update `docs/PLAN.md` to reflect current progress and the delay. Preserve teacher-vs-student comparison and evaluation; apply the specified cut order, present the revised plan, and wait for approval before continuing.
- **Wrap-up:** Run required checks, update decision/results/cheatsheet logs, commit each approved working item, and push the completed commits.
