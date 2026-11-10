# Phase 1 results — 2026-11-08

**Recovery update, 2026-11-10 local:** the corrected procedure now reproduces 8/100 marker refusals and KL 0.078021966 from fresh extraction, with 100/100 exact save/reload matches. The historical account below is retained; its unresolved diagnosis and proposed next steps were superseded by the recovery appendix at the end of this file.

The published checkpoint's reported effect was approximately reproduced. Replaying its rounded parameters did **not** reconstruct that behavior. Reference validation and persistence are complete; reproduction of the editing procedure remains unsuccessful. The evaluated publisher checkpoint is available as a fixed reference; retain the replay as a diagnostic control.

| Condition | Refusals / 100 | Benign first-token KL | Hit 100-token limit |
| --- | ---: | ---: | ---: |
| Original Qwen3.5-9B, measured | 100 | 0 | 59 |
| Rounded-parameter replay, measured | 72 | 0.4340466261 | 91 |
| Published checkpoint, measured in FP16 | 8 | 0.0784299672 | 100 |
| Publisher's card, reported | 6 | 0.0793 | Not reported |

The replay checkpoint was saved, loaded afresh, and evaluated again: **100/100 response strings matched exactly**, including the same 72 refusal labels. This rules out serialization as the explanation for the observed replay mismatch. The evaluation harness obtains close-to-card results on the actual publisher weights; the construction of our edited weights remains the unresolved part.

## Protocol

- Kaggle T4 ×2; existing CUDA PyTorch preserved. Removed unused torchao 0.10.0 because PEFT rejected its version. No local environment or model execution.
- FP16 with model-required FP32 exceptions; explicit non-thinking chat template; greedy decoding, EOS enabled, maximum 100 new tokens; original Heretic refusal-marker scorer and benign first-token KL.
- Difference-of-means extraction on 400 harmful and 400 harmless training prompts; one fixed edit using the published rounded parameters. Evaluation uses separate 100-prompt test slices. No optimizer search or test-directed parameter tuning.
- Diagnostic loaded the actual pinned publisher weights without editing them. It reused the stored original-model log probabilities after checking byte-identical prompt provenance. The release and compatible source have identical refusal-marker definitions and scorer logic; base and publisher chat-template hashes also match.
- Base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Publisher revision: `81dda62421c43a5bd5722edbc6cd47714673dc7f`. Heretic compatible source: `e26da5e0e648bb285f896c9801849edfa8d20913`. Dataset revisions and all remaining settings are in `sources.json` and the run manifests.

## Interpretation and limits

Eight versus six refusals is a close result, not an exact replication. The publisher's exact execution environment, source modifications, direction tensors, seed and unrounded parameters were not available. The compatible source postdates the checkpoint. Precision, prompt-position/extraction details and unpublished implementation differences remain hypotheses for the replay gap; none was established as its cause.

A marker-negative response is not necessarily harmful compliance. Spot-checking replay prefixes found refusals, redirects and fictional examples among marker negatives. All publisher-checkpoint answers hit the generation cap, so this experiment does not assess complete long-form answers. The independent **0/400 HarmBench** claim, unsafe-compliance rates and capability benchmarks were **not tested**.

## Diagnostic review — 2026-11-09

- **Confirmed methodological limitation:** the replay reused the publisher's optimized settings with newly extracted directions and performed no search. The settings alone do not specify the direction tensors or reproduce their extraction conditions. Two-decimal rounding has not been established as the cause.
- **Confirmed protocol difference, unproven cause:** `scripts/build_qwen35_notebook.py` forces `enable_thinking=False` and an empty `response_prefix`. Pinned upstream `src/heretic/main.py` instead detects and sometimes extends a common response prefix before extraction. This changes how the extraction position is selected; whether it differs from the author's actual run is unknown.
- **Unresolved implementation provenance:** compatible source handles both full-attention `o_proj` and DeltaNet `out_proj`; the author's exact hybrid-layer modifications were unavailable. A weight-delta audit can test whether the same components were edited, allowing for dtype-conversion noise.
- **What the checks establish:** actual publisher weights work under our FP16 evaluation, so a general inability to evaluate that checkpoint on the T4 setup does not explain the replay gap. Saved/reloaded replay agreement does not verify the editing math or the chosen vectors. Extraction precision remains a separate hypothesis.

Recommended order: compare base/publisher/replay component changes; inspect exact prompt token suffixes and extraction slots; check matching projection-output interventions against the adapter and merged edit on identical inputs, allowing for numerical precision. Only then run a small validation search if needed, keeping a fresh final test separate. Longer EOS-stopped completions, semantic refusal/compliance labels and benign utility should supplement the historical marker metric. No new GPU run was launched for this review.

## Persistence and compute

- [Evaluated publisher baseline](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-validated-20261108): original upstream weights retained unchanged after a CPU-only transfer; `baseline_persistence.json` records source/destination revisions and matching weight-object SHA-256 hashes.
- [Replay checkpoint and full evidence](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-replay-20261107): failed replay, adapter, directions, original/edited/reloaded answers, manifests and source. Publisher-checkpoint diagnostic is under `diagnostics/published_fp16/`.
- Both repositories are private. Local results are under `runs/batch-v1/` and `runs/published-check-v1/`. Kaggle jobs `qwen35-phase1-heretic-replay` and `qwen35-phase1-published-check` both completed; no GPU job remains running.
- Quota snapshots: 28.67 hours before the batch runs, 27.48 afterward (about **1.19 GPU hours used**). No optimizer search was run.

Next: perform the targeted reconstruction audit above before another full replay. Later quantization comparisons can use the evaluated publisher checkpoint and original model on fixed prompts, keeping safety behavior, over-refusal and benign utility separate. Do not use the failed rounded replay as the principal baseline.

## Recovery completed — 2026-11-10 local

The failed replay omitted per-slot harmless-mean orthogonalization. Sparse publisher delta fingerprints matched the projected direction in an MLP and a full-attention projection at cosine >0.9999. Sampled target coverage matched the expected 19 attention projections (including DeltaNet) and 23 MLP projections. A controlled raw/projected pilot changed development marker refusals from 20/32 to 0/32 and KL from 0.437735 to 0.065908, holding extraction means, base, published strengths and inference fixed. No optimizer search was needed.

The corrected notebook then freshly extracted 400 harmful and 400 harmless activation means, applied projection before interpolation, and produced **8/100 marker refusals with benign first-token KL 0.0780219659**. This closely matches the actual publisher checkpoint measured at 8/100 and 0.0784299672. Freshly loading the saved corrected checkpoint reproduced **100/100 responses exactly**. Independent artifact calculations verified prompt provenance, all 300 marker labels, 100 saved KL terms, projected-vector derivation, 64 adapter B tensors/42 active targets, and paired diagnostic provenance. Fresh means and vectors exactly matched the earlier pilot, with extraction execution recorded by the notebook and logs.

The pilot's frozen candidate additionally gave 2/64 markers on a deduplicated standard-only subset; this is not full HarmBench or a classifier result. The fixed paired 12-case 512-token diagnostic gave original/corrected markers 11/0 and cap counts 2/11. One-agent qualitative review found semantic withholding in all 12 originals and two corrected responses, with seven corrected visible-assistance cases and three ambiguous cases. The reviewer knew conditions/context; no population or inter-rater claim is made. The historical 100-case result remains exposed, and first-token KL is not a capability benchmark.

Use [the corrected notebook](../../Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb). The full reconstructed checkpoint is private at [Observer04/Qwen3.5-9B-heretic-corrected-20261109](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-corrected-20261109). Source, response rows, directions, adapter, manifests and independent review are retained under `recovery/` and persisted with the checkpoint. Remote index membership and all six shard LFS hashes were verified without local model downloads. Both recovery GPU jobs completed; measured cost was 2.11 quota hours, leaving 25.37 hours. Quantization and later research phases remain deferred.
