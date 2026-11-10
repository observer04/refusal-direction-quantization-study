# Third review: a conditional research program, not a paper claim

**Assessment date: 10 November 2026.** Independent adversarial review of recommendations A/B, `research_idea.md`, and the recovery evidence summarized in `STATE.md`. Research/design only; no model run or quota expenditure.

## Verdict

**Choose a narrower version of A for a small pilot. Keep B as a diagnostic, not the main contribution. Neither current proposal establishes major-conference novelty.** A can become useful research if it explains and predicts *which behavioral edits fail during deployment conversion*, and identifies a correction that beats straightforward alternatives. Merely observing quantization damage, direction drift, edit-order differences, or repaired refusals is already crowded territory.

The corrected Qwen result—8/100 marker refusals, first-token KL 0.07802, and identical reload responses—establishes reproducibility of an existing method. It does not establish semantic compliance, preserved capability, or novelty. The 99/100 capped completions make a sound semantic evaluation prerequisite to interpreting another transformation.

A's identity `E = Q(W+Δ) − Q(W) − Δ` defines the edit error; nonzero error follows from nonlinear rounding. Removing `rrᵀE` removes that component by construction. Neither is a discovery. B's refreshed vector being orthogonal to the quantized benign centroid is also constructed, while centroid protection already appears in [biprojected abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration). B additionally risks trading away intervention strength: lower benign KL can result from a weaker edit. One mean protects neither individual benign examples nor each target writer's inputs.

The most serious new prior-art check is practical: [OrcaRouter's GLM FP8 card](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) already reports projection leakage, lossless-versus-baked behavioral differences, and iterative repair. A [Qwen NVFP4 card](https://huggingface.co/lflare/Qwen3.8-Flash-Next-NVFP4-Abliterated) already uses constrained requantization to preserve a refusal projection. These are self-reported model cards with limited scoring, but they invalidate a claim that discovering or correcting requantization leakage is itself new.

## One exact question

**Can edit-conditioned output error predict, and causally explain, the loss of a fixed behavioral edit during weight quantization better than ordinary reconstruction error or direction similarity?**

Falsifiable hypothesis: with the base, edit, quantizer grid, and prompts fixed, failure is disproportionately carried by the rounding error projected onto the edit's output subspace. Cancelling that component will preserve the intended semantic effect better than matched random corrections and a generic low-rank reconstruction correction, without greater benign damage. A later prediction rule must work on untouched model/quantizer conditions; fitting a correlation retrospectively is insufficient.

The deployment implication is a decision about whether a behaviorally tested edit can be baked into a compressed checkpoint, needs a small floating adapter, or needs revalidation. This matters for safety-enhancing edits as well as safety removal. If true, checkpoint conversion could be assessed using an edit-specific diagnostic rather than benchmark accuracy alone. If false, representation drift, distributed errors, or prompt dependence—not a simple projection leak—require a different account.

**Novelty confidence: low-to-moderate for this exact causal/predictive increment; low for a new repair method.** This bounded search through the assessment date cannot certify absence of closer work. The evidence table identifies the actual remaining increment.

## Affordable first experiment

Use the saved original and corrected weights, actual `Δ`, saved direction, exact writer manifest, strengths, and chat template. Freeze one base-derived quantizer's codebook/scales/groups/exclusions. Fix KV precision. Do not re-extract directions. If frozen-grid packing cannot be implemented faithfully, label the result a dequantized diagnostic; do not present it as production low-bit inference.

Define `Q` as quantize/dequantize with that frozen configuration. Compare `W`, `W+Δ`, `Q(W)`, `Q(W+Δ)`, and `Q(W)+Δ`. The last is an unmerged floating-edit counterfactual, not a storage-matched deployment alternative. At every edited writer compute `E` above and `P=rrᵀ` in output coordinates. Add four diagnostic conditions on `Q(W+Δ)`: subtract `PE`; subtract each of two fixed-seed, rank/energy-matched random-axis corrections; and apply a generic activation-weighted rank-one reconstruction correction with the same parameter budget. The generic comparator reconstructs total quantization error, using benign calibration only. Also verify zero-edit identity, correct tensor orientation, and cached-input numerical equivalence. Native refitted scales are a separate later experiment.

Start with at most **32 new, disjoint prompts**: 16 harmful, 8 sensitive-but-benign, 8 utility. Nine conditions give at most 288 completions. A hard **2 Kaggle quota-hour exploratory budget**, including loading, conversion, capture, generation and any GPU judging, is a proposed spending limit—not a runtime estimate. Profile first; reduce the balanced prompt block or stop if needed. Never compare arms on different selected prompts. Saved means are insufficient for per-example noise estimates or activation-weighted corrections; record new, small calibration captures explicitly. This pilot can reveal a large failure or implementation problem; it cannot establish preservation or a ten-point effect reliably.

Use paired prompt-level estimands. For harmful-assistance rate `H`, the primary deployment interaction is

`I = [H(Q(W+Δ)) − H(Q(W))] − [H(W+Δ) − H(W)]`.

The floating-edit comparison isolates baking error on the same quantized base. Correction contrasts test whether the selected error component matters behaviorally. Report semantic refusal and harmful assistance separately, plus benign over-refusal, utility correctness, degeneration, token count and censoring. Blind and randomize response labels; use a fixed rubric and independently double-label a subset. Markers remain diagnostics. Judge only visible assistance; classify unfinished outcomes as unresolved and report best/worst-case bounds. EOS stopping, equal caps and explicit truncation rates are mandatory; capped outputs must not become successes. Report paired uncertainty, denominators, and all arms. First-token KL is supplementary, not capability evidence. Select any strength or diagnostic threshold on development data only, then freeze it. [Iso-effect steering work](https://arxiv.org/html/2609.06473v1) already shows why matching behavioral effect and accounting for termination failures matter.

## What would justify a larger paper

1. **Establish the mechanism locally.** Show a material semantic interaction; distinguish ordinary base damage from baking error; demonstrate targeted correction beats shams and generic reconstruction. Measure induced writer-output error on fresh inputs, rather than only weight leakage. A correction winning because it changes edit strength does not establish selectivity.
2. **Derive an actionable prediction and test it independently.** A proposed input-weighted subspace-error diagnostic must predict effect loss beyond weight MSE, perplexity, direction cosine and static projection leakage. Freeze it before testing a held-out model/quantizer combination. Failure to generalize is evidence against the proposed principle.
3. **Expand only after that signal.** Confirmation needs an independent dense transformer alongside Qwen's hybrid architecture; multiple weight quantizers with different calibration/rounding mechanisms; separate native-scale conversion; and distinct prompt domains, including benign transfer. Include safety-enhancing and an unrelated behavioral edit to establish scope. Compare constrained requantization, stochastic rounding and generic error repair at matched storage/latency budgets. This requires future compute, not an assumption that the remaining 25.37 hours covers it.
4. **Release an auditable artifact.** Pin revisions, packing metadata, writer coverage, prompts/splits, full outputs, labels, censoring and resource accounting; provide original/edit hashes and a separate untouched confirmation. A reusable diagnostic plus a causal mechanism or a demonstrably better decision/repair rule could support a credible AAAI/top-tier submission. More benchmarks around one checkpoint, a pretty cosine plot, or a tiny MMLU difference would still be insufficient. Acceptance cannot be promised.

## Kill or pivot criteria

- If semantic evaluation cannot separate genuine assistance from redirects/truncation, finish measurement before extending the experiment.
- If interaction is practically small with a suitably narrow interval, report robustness for these conditions; do not search harder quantizers until something breaks. A wide interval means underpowered, not equivalent.
- If random/generic corrections match the targeted one, drop the refusal-specific mechanism claim. If existing constrained requantization solves the problem at equal cost, document that engineering result rather than rename it.
- If weight projection is restored but behavior is not, report the dissociation and investigate distributed/application effects. Do not force B to become the explanation.
- If B is explored, it must beat matched rotations and preserve utility at matched semantic effect on a new benign domain. Orthogonality, lower KL, or lower refusal wording alone is not success.

These are scientific decision criteria, not implementation gates. Null, negative and inconclusive outcomes remain valid reports when uncertainty and scope are explicit.

## Compact primary-source evidence

| Closest prior | Status / verified coverage | Increment we would still need |
|---|---|---|
| [Zhang et al., QERA](https://proceedings.iclr.cc/paper_files/paper/2025/file/21718991f6acf19a42376b5c7a8668c5-Paper-Conference.pdf) | ICLR 2025; low-rank high-precision correction with activation-weighted reconstruction. Official slides also specify the objective. Full-paper fetch timed out here. | Edit-conditioned behavioral attribution and prediction beyond generic reconstruction, not “a low-rank quantization correction.” |
| [Chen et al., Q-resafe](https://proceedings.mlr.press/v267/chen25ci.html) | ICML 2025; full methods inspected. DPO patching, quantized LoRA updates and periodically identified safety-critical weights; broad quantizer/calibration safety evaluation. | Frozen-edit transport mechanism without retraining, with causal counterfactuals; generic safety patching is established. |
| [Xu et al., Channel Geometry of Refusal](https://openreview.net/pdf?id=n1CfsXNCSL) | ICML 2026 mechanistic-interpretability **workshop**. Indexed methods and held-out appendix inspected: fixed-weight KV quantize/dequantize, layer/channel diagnostic, mitigation and unseen-model/prompt tests. Direct PDF access challenged/403. | Weight-edit rounding interaction, not another general refusal-geometry diagnostic/repair story. KV and weight quantization must not be conflated. |
| [Lai, Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) | Author's 2025 community article; centroid constraints, target-layer protection, and quantized activation measurement/full-weight assembly. Not peer reviewed. | B needs evidence that benign-only precision refresh adds selective, cross-domain value beyond these constructions and weaker-edit controls. |
| [OrcaRouter, GLM FP8 edit](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) | Practitioner model card: lossless writer hooks versus baked weights; leakage reduction with more refinement; opening-phrase scoring explicitly limited. | Controlled causal attribution, semantic measurement and predictive generalization beyond one deployment demonstration. |
| [lflare, Qwen NVFP4 edit](https://huggingface.co/lflare/Qwen3.8-Flash-Next-NVFP4-Abliterated) | Practitioner card: quantized extraction, grid-level edit loss, constrained code changes to preserve projection, protocol/scorer caveats. | A must distinguish itself from existing constraint-preserving compilation and beat it or explain its failures. |

Additional measurement constraint: [Varshney's noise-floor preprint](https://arxiv.org/html/2609.30275v1) already studies within-precision split-half nulls and separates translation/attenuation; its causal-transfer rates are explicitly withheld because semantic labels are unvalidated. The page's August date conflicts with its September-form identifier, so chronology needs checking. It prevents presenting estimator-noise controls or cosine caveats as new. All model-card results above are reported claims, not independently reproduced evidence.
