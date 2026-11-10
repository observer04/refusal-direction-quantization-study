# Independent recommendation B

Research/design review only. Written 2026-11-10; no new model execution. I read `research_idea.md` and `STATE.md` independently of reviewer A.

**Primary recommendation: test whether the harmless-mean constraint survives compression. Fallback: isolate quantization of the weight edit from transfer of its direction.** Both are bounded mechanism tests; neither warrants a new-method claim yet.

## What the existing run establishes

The corrected Qwen3.5-9B reconstruction reached 8/100 marker refusals and first-token benign KL 0.07802; its saved/reloaded outputs match. The raw-direction failure was omission of an existing projection step, not a discovery. Historical prompts were exposed, most responses were truncated, and first-token KL does not establish capability retention. Use this checkpoint and recipe as a fixed reference, with fresh evaluation prompts.

The proposal's broad novelty claim needs revision. Refusal-vector drift, its estimator noise, and FP16-vector transfer are already studied by [Varshney](https://arxiv.org/html/2609.30275v1). Quantized steering dose-response and capability costs are studied by [Bhandari and Wade](https://arxiv.org/html/2609.06473v1). Merely changing their target to refusal would be weak novelty. Quantization/safety interaction is established by [Chen et al.](https://proceedings.mlr.press/v267/chen25ci.html).

Bibliographic check: the proposal's [2502.09674](https://arxiv.org/abs/2502.09674), [2606.22686](https://arxiv.org/abs/2606.22686), and [2606.25750](https://arxiv.org/abs/2606.25750) resolve to multidimensional refusal, contrastive **logit** steering, and RAS/SafeVec respectively. They support adjacent questions; they do not establish our proposed interaction's novelty or prove architecture determines safety. Its sample cosine numbers are hypothetical. The original order comparison also changes the extracted vector, so it cannot isolate order.

## 1. Primary: does compression invalidate harmless-mean protection?

**Hypothesis and estimand.** Re-estimating only the benign constraint after compression improves selectivity of a transferred FP16 edit beyond an equally sized, arbitrary rotation. Estimate the paired change in benign distortion and task accuracy between refreshed and frozen directions, alongside the change in semantic harmful assistance. A reduction in refusal alone is not success.

For each existing extraction slot, let `dF` be its frozen FP16 difference of means, `gF/gQ` its benign means before/after quantization, and `P(g)` projection orthogonal to `g`. Compare:

- **Frozen transfer:** `rF = normalize(P(gF) dF)`.
- **Benign-only refresh:** `rR = normalize(P(gQ) rF)`; no new harmful mean.
- **Full re-extraction:** `rQ = normalize(P(gQ) dQ)`; a separate comparator, not the same intervention.

Apply these operations per slot **before** the existing slot interpolation; keep target modules and component strengths fixed. Refresh is a minimal further rotation, not equivalent to recomputing the published formula. That distinction is intentional. [Lai's projected and biprojected methods](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration) already protect harmless centroids across layers. The incremental question is whether **precision-induced loss of that constraint**, rather than a changed refusal estimate, explains collateral effects. I did not verify an exact preceding experiment; this is provisional, not proof of novelty.

**Minimal controlled design.** Start with FP16 and one NF4 weight-only configuration. Use the same packed quantized base and unmerged FP16 rank-one updates for all quantized arms; compute each update against that base's dequantized weights. Freeze source slots, interpolation, module list, chat template, greedy non-thinking generation and EOS handling. Include zero-edit baselines at both precisions. Primary comparisons use the fixed published strength; a small development-only strength grid can additionally compare utility at matched semantic assistance levels, borrowing the existing iso-effect framework. Never tune on the final set.

**Essential controls.** Include two sham rotations matched to the refresh angle and approximately matched edit energy; report the remaining energy difference. Repeat benign-only refresh using two disjoint benign calibration halves. Measure within-precision split-half variation before calling a centroid/vector change “drift.” Existing artifacts contain means, **not per-prompt activations**: paired FP16/NF4 captures for resampling require new Kaggle work. Use exactly the original ordered extraction prompts at both precisions, cache relevant per-prompt slot states, and keep task/domain mix fixed. Full re-extraction additionally needs a quantized harmful-set pass. Evaluate on other benign domains too. Centroid orthogonality protects neither individual prompts nor every target layer; difference-of-means may mix harmful intent with refusal.

There is no merging/requantization in this primary contrast. NF4 is a nonuniform 4-bit codebook with FP16 compute, **not integer INT4**; freeze block size, double-quantization setting, exclusions, library revisions and device map. NF4 has no text calibration fit; the centroid-estimation set is a different object. If later using GPTQ/AWQ, independently hold their calibration text/order and quantizer fitting fixed. Hybrid kernels/module coverage must be verified; a fallback to FP16 modules must be reported.

**Informative null and limits.** No excess benign damage, or refresh indistinguishable from shams, bounds this explanation for this configuration. Refresh that reduces assistance proportionally to benign distortion gives no demonstrated selectivity gain. A positive result supports a compression-sensitive constraint in this Qwen recipe, not a universal safety subspace or a cross-architecture claim. Novelty confidence: **moderate-low**, because centroid protection and noise-aware transfer are existing ideas.

## 2. Fallback: is a deployment gap caused by rounding the edit?

**Hypothesis and estimand.** At fixed `r` and strength, quantizing a merged edit changes its realized function more than retaining the corresponding edit in FP16 on the same quantized base. Estimate paired semantic/utility differences and the error in the realized component update. Practical runtime projections and quantized-checkpoint edits already exist; the contribution would be controlled attribution, not discovering noncommutativity.

For each edited output matrix define `T(W) = (I − λrrᵀ)W`, and `Z(W)=dequantize(quantize(W))`. Compare `Z(W)`, `Z(T(W))`, and `T(Z(W))`; the last retains a high-precision correction and therefore has a different storage budget. To compare **two quantized storage paths**, additionally use `Z(T(Z(W)))`, with the no-edit round-trip `Z(Z(W))`. Hold `r`, strengths and target matrices fixed: re-extraction or reoptimization answers another question.

Use identical quantizer settings. Report native scale refitting separately from a frozen-codebook/scale diagnostic; the latter is a mechanism control, not ordinary deployment quantization. Include matched random-direction edits, zero strength, and FP16 hook-versus-merged equivalence. Compare component outputs on cached identical inputs as well as complete generations. Since published strengths can exceed one, `rᵀT(W)` need not vanish: measure error relative to the intended update, not a presumed zero residual.

Use the evaluation below. A null with useful uncertainty supports edit retention in this configuration; a gap explained equally by random edits supports generic low-rank rounding, not refusal-specific damage. Confounds include different high-precision budgets, fused kernels and scale refitting. Honest claim: a materialization study of this reconstruction. Novelty confidence: **low-to-moderate**; safer as a fallback diagnostic.

## Shared evaluation and compute envelope

Freeze deduplicated development and final subsets separately from extraction. Report **semantic withholding, useful harmful assistance, benign over-refusal, and ordinary task accuracy separately**, plus marker disagreement, incoherence and token-cap censoring. Use a frozen StrongREJECT-style rubric/judge with blinded independent human checks; use XSTest safe cases for over-refusal and a small fixed reasoning/instruction-following set for utility. Teacher-force a fixed benign continuation set for multi-token KL; retain historical first-token KL only for continuity. Pair comparisons by prompt and report bootstrap intervals and disagreements. Longer identical generation caps matter because our edited outputs often hit 100 tokens.

Budget one study, not both: at most **1.5 reported Kaggle quota-hours** for loading, module verification, calibration capture and a small throughput pilot, then a provisional **8-hour total ceiling** including the pilot. Remaining quota was 25.37 hours at review time; actual throughput and accounting determine affordable sample sizes. Candidate final size: 100 harmful, 50 over-refusal, 100 utility cases, plus teacher-forced benign KL. Reduce arms or report wider uncertainty if this exceeds measured capacity. Do not promise equivalence from these small samples. Full benchmark sweeps, second models, multiple quantizers and optimization searches are outside this design. A clear null is a reportable answer, not a reason to keep searching for an effect.

## Compact primary-source map

Dates are source-reported; repositories are practical precedents, not peer-reviewed evidence. Coverage gaps below concern the inspected material, not an exhaustive novelty certification.

| Primary source / date | Already covers | Does not establish our narrower claim |
| --- | --- | --- |
| [Arditi et al., Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717), 2024-06-17 | Directional activation ablation and permanent weight intervention. | Harmless-constraint transfer under compression. |
| [Lai, Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration), 2025-10-25 | Removing the harmless-mean component before ablation. | Compression-specific causal benefit of refreshing that constraint. |
| [Lai, Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration), 2025-11-06 | Target-layer harmless protection and norm preservation; reports returning refusal. | Selectivity after precision changes; centroid protection is not our invention. |
| [Chen et al., Assessing Safety Risks and Quantization-aware Safety Patching](https://proceedings.mlr.press/v267/chen25ci.html), ICML 2025 | Safety changes under quantization and restoration. | Our isolated benign-constraint contrast. |
| [Varshney, Cosine Similarity Is Not Evidence](https://arxiv.org/html/2609.30275v1), 2026 | Refusal-vector transfer, estimator floors, within-precision nulls and translation/attenuation ambiguity. | Benign-only constraint refresh as a selectivity intervention. Page displays 12 August despite identifier 2609; month not silently inferred. |
| [Bhandari & Wade, Steering Under Compression](https://arxiv.org/abs/2609.06473), 2026-09-06 | INT8/NF4 steering, iso-effect capability costs, dose-response/censoring; sentiment and reasoning length. | A causal decomposition of centroid protection versus full refusal re-extraction. |
| [Su & Zhang, Structural Collapse of Safety-Critical Weight Protection Scorers](https://aclanthology.org/2026.findings-emnlp.2116/), 2026-10 | Protected-block rankings; restored refusal can coexist with poor capability, robustness and over-refusal. | Harmless-centroid refresh; forbids claiming refusal restoration alone restores safety. |
| [Qwen3.8 runtime projection author repository](https://github.com/pocharlies-org/qwen38-27b-rank1-refusal-projection), dated 2026-08-29 section; [HGN edit/requantization author repository](https://github.com/dowjames/hgn-abliterate), accessed 2026-11-10 | Practical quantized runtime intervention and dequantize/edit/requantize workflows. | Controlled evidence that their materialization gaps are refusal-specific. Small author benchmarks are not independent guarantees. |
| [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314), 2023-05-23 | NF4 and quantized bases with higher-precision low-rank adapters. | A new adapter/quantization method here; NF4 is not integer INT4. |
| [Souly et al., StrongREJECT](https://arxiv.org/abs/2402.10260), 2024-02-15; [Röttger et al., XSTest](https://arxiv.org/abs/2308.01263), 2023-08-02 | Useful harmful assistance versus empty jailbreaks; benign over-refusal. | Population-level conclusions from our historical marker counts. |
