# Research direction after three reviews

Decision date: 2026-11-10. Literature/design recommendation only; no new experiments are authorized or running. [Review A](recommendation_a.md) and [Review B](recommendation_b.md) were written independently. [Review C](recommendation_c.md) then challenged both and their closest prior work.

## Recommendation

**Pursue a small feasibility study of A's causal attribution question. Keep B's harmless-centroid refresh as a secondary diagnostic. Neither currently justifies a major-conference novelty claim.**

The exact question is:

> Can edit-specific quantization error predict which behavioral weight edits lose their intended effect, and can a correction at the same parameter budget outperform generic reconstruction or existing projection-preserving conversion?

This could matter whenever a tested model edit is converted for deployment, including safety-enhancing edits. The practical outcome would be a validated way to decide whether an edit can be merged into compressed weights, needs a small floating adapter, or requires fresh behavioral evaluation. Generalizing beyond refusal suppression would make the contribution more useful.

## Where the novelty would have to be

Quantized refusal mechanisms, direction transfer and noise-aware geometry already have prior work; see the source tables in all three reviews. Practitioner [Qwen NVFP4 documentation](https://huggingface.co/lflare/Qwen3.8-Flash-Next-NVFP4-Abliterated) already reports edit loss from rounding and constrained requantization. These are reported model-card findings, not independently reproduced results. Low-rank quantization-error repair also exists, including [QERA](https://proceedings.iclr.cc/paper_files/paper/2025/file/21718991f6acf19a42376b5c7a8668c5-Paper-Conference.pdf).

Consequently, a nonzero rounding error, restored orthogonality, different operation orders or a better refusal score cannot carry the novelty claim. The possible increment is **causal behavioral attribution plus independently validated prediction**, with targeted correction outperforming matched controls and relevant existing methods. Novelty confidence remains low-to-moderate for that narrow increment; the search is bounded and does not certify priority.

## First experiment

First establish semantic behavior and benign utility for the original and corrected checkpoints; our historical marker counts and first-token KL are insufficient.

Then freeze the actual weight edit, vectors, writer modules, strengths, prompts and one quantizer configuration. Compare original/edited FP16 weights, original/edited quantized weights, and the floating edit retained on the same quantized backbone. Separate fixed-grid diagnostics from ordinary conversion that refits scales. Re-extraction or optimization belongs in a separate experiment.

Use edit-induced output error to construct one targeted correction. Compare it against two rank/energy-matched random corrections and a generic activation-weighted reconstruction correction. Check numerical equivalence before interpreting generations. Budget and storage differences of a floating adapter must be reported.

The proposed exploratory ceiling is **32 balanced prompts × 9 conditions, at most 2 Kaggle quota-hours including setup and GPU scoring**. Profile first; this is a spending cap, not a runtime prediction or a powered final test. New per-input activation capture is needed; our saved means cannot supply it. Score semantic refusal, useful harmful assistance, over-refusal, task correctness and truncation separately, using paired uncertainty and blinded review. If packing or evaluation is unreliable, resolve that before interpreting a mechanism.

## Path to a larger paper

A credible major-conference submission would need all of the following:

1. A causal result that distinguishes general quantization damage from loss of the particular edit, and beats sham and generic corrections at matched budgets.
2. A diagnostic or principle frozen before confirmation, predicting failures on held-out model/quantizer conditions beyond weight error, cosine similarity and existing measures.
3. Independent confirmation in another model family, multiple quantizers, fresh prompt domains, and safety-enhancing or unrelated behavioral edits. This requires future compute; the current quota does not guarantee coverage.
4. A reproducible artifact and an actionable outcome, with semantic/utility trade-offs and conversion settings documented.

A small or uncertain interaction, corrections that work equally well for random directions, or an existing converter solving the problem would weaken the proposed contribution. Report those outcomes honestly; do not switch quantizers or prompts until a preferred result appears. A null can support a scoped undergraduate report without supporting a general mechanism or conference-level claim.

**Next action:** prepare the semantic baseline and the bounded feasibility protocol. Do not launch a broad quantization sweep or commit the report to a novelty claim yet.
