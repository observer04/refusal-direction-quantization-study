# Independent model audit v1

Start with [AUDIT.md](AUDIT.md). It separates weight correctness, phrase counts, semantic behavior, the actual publisher comparison and the external 0/400 claim.

- [Item-level W/D/P review](ITEM_REVIEW.md), [paired full outputs](HUMAN_REVIEW_PAIRED.html), [condition-masked human review](HUMAN_REVIEW_BLINDED.html).
- [Frozen protocol](PROTOCOL.md), [rubric](review_rubric.md), [decision calculations](decision_rule_results.json), [semantic counts](semantic_summary.json).
- [Exact detector comparison](classifier_comparison_all.csv), [source audit](SOURCE_METRIC_AUDIT.md), [minimal preprocessing witness](classifier_preprocessing_witness.json).
- [Run evidence](runs/run-v1/output/model_audit_v1/results), [integrity checks](verification/run_integrity.json), [artifact QA](verification/final_artifact_verification.json), [auxiliary hash exception](verification/auxiliary_listing_hash_discrepancy.json).

Human fields are blank; agent labels are separate and frozen. Browser rating controls do not autosave. All 48 new and 24 reused outputs are retained. No further generation is planned. Private persistence is documented by the local `persistence_verification.json` after upload.
