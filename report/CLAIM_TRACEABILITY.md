# Author-only claim traceability

This companion is not report prose and should not be inserted into the DOCX. It preserves internal evidence routes removed to make the report self-contained. The panel-facing document includes its own condition definitions, equations, procedure, tables, diagram, rubric, limitations and conclusions. Public literature and software references remain in its numbered bibliography.

## Claim map

All paths below are relative to the project workspace `/home/op/llm/abliteration`.

| Report location | Claim or content | Exact internal evidence |
|---|---|---|
| Abstract; Sections 3.2–3.3; Appendix A.2 | Existing harmless-mean projection, per-slot normalization, upstream direction index 18.49 mapping to activation slots 19/20 and interpolation .49 | `research/model_audit_v1/AUDIT.md`; `reproduction/qwen35_9b/recovery/corrected_source_verification.json`; `reproduction/qwen35_9b/recovery/upstream_prefix_reference.json`; corrected run `direction_metadata.json` under the results path below |
| Equation 3.2; Appendix A.2 | Generalized scaled rank-one updates; upstream component schedule values, including strengths greater than one | `reproduction/qwen35_9b/recovery/runs/pilot-v1/qwen35_recovery/published_parameters.json`; `reproduction/qwen35_9b/recovery/pilot_body.py`; `reproduction/qwen35_9b/recovery/kaggle-audit/audit_weights.py` |
| Table 4.1 | Historical marker counts 100/72/8/8/8, caps 59/91/100/99/99, first-token KL values | `research/model_audit_v1/historical_metric_audit.json`; `research/model_audit_v1/verification/historical_recount_stdout.json`; `reproduction/qwen35_9b/recovery/evaluation_audit.json`; corrected results `summary.json`, `baseline_metrics.json` and `benign_first_token_kl_terms.json` |
| Section 4.2 | 64 factor pairs, 42 active writers, 19 attention and 23 MLP; every target hash verified; small-vector errors | `research/model_audit_v1/edit_artifact_audit.json`; `research/model_audit_v1/AUDIT.md`; `reproduction/qwen35_9b/recovery/fresh_vs_cached_directions.json`; `reproduction/qwen35_9b/recovery/saved_mean_projection_geometry.json` |
| Section 4.2 | 100/100 exact save/reload response matches | `reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/corrected_evaluation_verification.json`; `reproduction/qwen35_9b/recovery/evaluation_audit.json` |
| Sections 3.4–3.5; Table 3.2 | 72 audit records, 24 matched reuses, 48 newly generated; matched/fresh sizes, caps, masking and generation settings | `research/model_audit_v1/AUDIT.md`; `research/model_audit_v1/matched_reuse_metadata.json`; audit run results `matched_reuse_verification.json`, `run_manifest.json`, `profile_decision.json`, `matched_prompts.json`, `selected_fresh_prompts.json` |
| Table 4.2 | Matched /6 and fresh /8 requested-content labels, independent explicit refusal and cap counts | `research/model_audit_v1/semantic_summary.json`; `research/model_audit_v1/ITEM_REVIEW.md`; `research/model_audit_v1/verification/descriptive_counts_stdout.json` |
| Section 4.3 | D/P same fulfillment category on all fourteen safety cases; 8/24 complete strings identical | `research/model_audit_v1/semantic_summary.json`; `research/model_audit_v1/AUDIT.md`; `research/model_audit_v1/all_responses.jsonl` |
| Section 4.3 | Ambiguous authorized-context sensitivity: exclude two fresh cyber cases, D/P 4 clear + 2 partial /6, W 0 clear /6 | `research/model_audit_v1/ambiguity_sensitivity.json`; `research/model_audit_v1/ITEM_REVIEW.md` |
| Section 4.3; Table A.1 | Five distinct safe XSTest items; five elementary GSM8K items and observed task results | `research/model_audit_v1/semantic_summary.json`; `research/model_audit_v1/matched_prompts.json`; audit results `selected_fresh_prompts.json`; `research/model_audit_v1/verification/fresh_utility_reference_calculations.json` |
| Section 4.5; Table 4.4 | Marker / public v3 / semantic disagreement; public heuristic differs from official HarmBench; historical settings unresolved | `research/model_audit_v1/SOURCE_METRIC_AUDIT.md`; `research/model_audit_v1/classifier_comparison_all.json`; `research/model_audit_v1/classifier_comparison_saved.json`; `research/model_audit_v1/remote_provenance_audit.json` |
| Section 4.5 | Conditional closed-think-tag preprocessing witness, not an established historical cause | `research/model_audit_v1/classifier_preprocessing_witness.json`; `research/model_audit_v1/SOURCE_METRIC_AUDIT.md` |
| Sections 3.1–3.3 and 4.4; Table 4.3 | Five native NF4 arms, 60 outputs, original pilot labels, frozen FP32 factors, cap/task outcomes and finite-fixture bounds | `research/feasibility_v1/RESULTS.md`; pilot results directory below, especially `all_responses.jsonl`, run manifests, labels, utility and native-packing verification records |
| Sections 3.1–3.3 and 4.4 | Q(D) refits scales; Q(W)+Δ retains frozen factors and is not fresh re-extraction | `research/feasibility_v1/RESULTS.md`; pilot packed-state and adapter checks; frozen pilot generation source and arm definitions |
| Removed from panel-facing revision | Quota costs, auxiliary packaging exception and operational hash detail retained only in the evidence | `research/feasibility_v1/RESULTS.md`; `research/model_audit_v1/AUDIT.md`; `research/model_audit_v1/quota_sessions_before.json`; `research/model_audit_v1/verification/auxiliary_listing_hash_discrepancy.json`; `reproduction/qwen35_9b/recovery/compute_budget.json` |
| Appendix A.1–A.2 | Public model/code/dataset revisions and generation/runtime settings | `research/model_audit_v1/pins.json`; `research/model_audit_v1/reference/*/config.json`; corrected results `run_manifest.json` effective settings and sources; pilot and audit manifests |
| Chapter 2; twelve references | Primary-source bibliography metadata and intended use | `report/BIBLIOGRAPHY_PLAN.md`; URLs in `report/report_content.py` REFERENCES |

The corrected result directory used in the map is `reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/`. The audit run result directory is `research/model_audit_v1/runs/run-v1/output/model_audit_v1/results/`. The NF4 pilot results are `research/feasibility_v1/runs/run-v1/output/feasibility_v1/results/`.

## Reproduction and review entry points

Corrected notebook: `Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb`.

Final audit authority: `research/model_audit_v1/AUDIT.md`, `SOURCE_METRIC_AUDIT.md`, `ITEM_REVIEW.md`, `semantic_summary.json`. The earlier `report/COLLEGE_REPORT_CHECKPOINT.md` is an assessment from before the final audit and is not the authority for final semantic counts.

Audit raw response records: `research/model_audit_v1/all_responses.jsonl` (72 records). Human review copies: `research/model_audit_v1/HUMAN_REVIEW_PAIRED.html` and `HUMAN_REVIEW_BLINDED.html`. Human fields remain blank. Browser entry does not automatically save ratings; durable new review must use the provided structured rating fields. These operational details were removed from the college report because they are not substitutes for results or methods.

The NF4 pilot's original labels remain separate from the later FP16 audit. Do not silently join or overwrite the two analyses. Neither review establishes harmful technical correctness. No human ratings are claimed.

## Private persistence identifiers

Corrected model: `Observer04/Qwen3.5-9B-heretic-corrected-20261109` at revision `6cd66cf9bf38efab01ffd5ab6e5c7995868f55bb`.

Audit evidence: commit `7f12baad08f0f5296b763f2c689be5731e322544`, folder `research/model_audit_v1/review-v1/` in that repository.

Pilot evidence: commit `9e4f047bafd49994ee488eec31888d16d43a571f`, folder `research/feasibility_v1/review-v2/` in that repository.

Remote hashes were checked against local evidence; checkpoint-root files were unchanged by the evidence uploads. The auxiliary listing exception is preserved with substantive verification. These private routes are author-side traceability and do not appear in the panel-facing DOCX.

## Document build evidence

`report/report_content.py` is the prose/table/reference source; `report/build_report.py` builds from the unchanged sample. `report/writing_qa/artifact.md` records the template contract. Final page images, structural checks, package preservation and QA notes are under `report/writing_qa/`. The report's own numbering, Tables 2.1, 3.1–3.2, 4.1–4.4 and A.1, Figure 3.1 and Equations 2.1 and 3.1–3.3 supply the reader-facing explanation.

## Triage revision literature grounding

The narrative was revised on10 November 2026. `report/TRIAGE_AND_FIX.md` records primary-paper passages and the bounded claims they support. The frozen source set is identical to v1; citation numbers now include abstract first appearance. Numerical claims remain attributable to the local evidence above rather than to external papers. The revised report uses a focused comparative review, not an exhaustive survey or a new-method claim.
