# Independent recommendation A

Research date: **10 November 2026**. This is a research design, not authorization to run it. I read `STATE.md`, the recovery findings, and the earlier AI-generated `research_idea.md`; its novelty assertions were not treated as evidence. I searched primary papers, author repositories and model cards for compression/refusal mechanisms, projected directions, cross-precision steering, editing order, quantizer calibration, and hybrid architectures. Thirteen search/open batches were attempted; one failed. Search coverage is bounded, not a systematic review.

## Decision

**Primary: causally separate edit-specific rounding from other quantization effects.** The potential contribution is a controlled attribution experiment on the recovered Qwen3.5-9B checkpoint. It is **not** the discovery that quantization affects safety, that directions survive quantization, that quantization and editing need not commute, or that hybrid models have refusal representations.

Those claims already have close precedents. [Chhabra and Khalili](https://arxiv.org/abs/2504.04215) test refusal mechanisms in quantized models; [Kadadekar](https://arxiv.org/abs/2606.10154) finds high direction cosine can coexist with changed safety; [Prosvirnin et al.](https://www.mdpi.com/2079-3197/14/10/261) evaluate steering in BF16 and NF4. [Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration), dated 25 October 2025, already compares full-weight and 4-bit direction geometry. Its harmless-mean projection is the recovered upstream step, not our invention.

Even edit survival is documented: the [HGN implementation](https://github.com/dowjames/hgn-abliterate) measures projection components reintroduced by requantization. A [primary model card](https://huggingface.co/TechPrototyper/Gemma4-31B-IT-PrismaQuant-6bit-abliterated-vllm) reports that apparent quantization damage disappeared under matched evaluation, and explicitly freezes production scales. These are useful practitioner evidence, not independently replicated findings. I did not find the exact **edit-induced error decomposition plus controlled causal correction** below; that is a provisional gap, not proof of priority.

## 1. Recommended primary question

**When quantization changes a fixed refusal edit's behavioral effect, is the change attributable to rounding introduced specifically by that edit along its direction, rather than baseline quantization damage or a newly estimated vector?**

Let `W` be the original weights, `W+Δ` the exact saved recovered edit, and `Q_S` dequantized weight-only quantization with the **same base-derived** scales, codebooks, groups and target manifest `S`. Do not assume every saved delta is rank one. First compare:

| Arm | Purpose |
| --- | --- |
| `W` | Original FP16 baseline |
| `W+Δ` | Edited FP16 baseline |
| `Q_S(W)` | Quantized original control |
| `Q_S(W+Δ)` | Quantized compiled edit |

For harmful-assistance rate `H`, preregister the interaction

`I_H = [H(Q_S(W+Δ)) − H(Q_S(W))] − [H(W+Δ) − H(W)]`.

Estimate the same contrast for semantic refusal, benign over-refusal and utility. This prevents a shift in the unedited quantized model from being mislabelled “loss of abliteration.” Preregister 10 percentage points as the primary practically important interaction; use paired prompt-bootstrap intervals. The pilot screens for a signal, and cannot establish equivalence unless its interval actually excludes that band. Size a subsequent held-out test from observed paired discordance, subject to quota.

For each edited matrix define

`E_edit = [Q_S(W+Δ) − Q_S(W)] − Δ`.

With the **frozen recovered** unit direction `r` at that matrix, set `P=rrᵀ`. Test a diagnostic arm `Q_S(W+Δ) − P E_edit`, implemented as an exact floating rank-one residual at the same Linear output. This removes only the edit-induced rounding component parallel to `r`. Compare zero-residual execution, a norm/rank-matched random-direction correction, and (if feasible) the complementary correction. Audit matrix reconstruction first. The correction is a mixed-precision counterfactual, **not** an equally compressed deployment artifact or a proposed novel quantizer.

An additional useful counterfactual is `Q_S(W)+Δ`, retaining the entire floating edit on a fixed quantized backbone. A gap from the compiled arm identifies compilation/rounding sensitivity; it does not establish a changed refusal mechanism. Re-extracting vectors is a **separate** extension: compare frozen `r_FP16` and re-extracted `r_Q` on the *same* quantized original, identical application sites and strength. Never change extraction and compilation simultaneously.

**Minimum experiment and cap.** Verify one quantizer's Qwen module coverage, FP16 compute on T4, unchanged tensor round-trip and exact residual arithmetic before generation. Begin with one 4-bit weight-only configuration; no precision sweep. Use 48 new fixed prompts: 24 harmful, 12 sensitive-but-benign, 12 utility. Four main arms require 192 target-model completions; reserve 64 for diagnostic corrections on a prespecified 32-prompt subset. Stop at **256 target-model completions or 2 Kaggle-reported quota-hours**, whichever comes first. The quota cap includes setup, loading, quantization, any extraction, target generation and GPU-based classification/scoring. Count classifier throughput separately; it does not acquire an additional GPU allowance. Reserve scoring time before launching all generations; leave an explicitly incomplete pilot if the cap is reached. Human review consumes no Kaggle quota but its elapsed effort must also be reported. Actual throughput and memory must be measured; this is a budget, not a runtime prediction. If a real backend cannot expose frozen quantizer state, use explicitly labelled dequantized/fake-quant diagnostic weights or stop; do not claim a matched production-quantizer result.

**Evidence and null.** Support requires a behavioral interaction plus a direction-specific correction changing it beyond random correction, without equivalent utility collapse. Stable cosine alone is insufficient. An informative null is no practically important interaction with intervals narrow enough to exclude the preregistered effect, or an interaction that parallel correction cannot explain. The latter redirects attention toward orthogonal errors, nonlinear propagation or calibration. Independently recalibrated quantization is a later **pipeline** comparison, clearly separated from fixed-grid rounding.

**Legitimate report claim:** “For this checkpoint, quantizer and prompt distribution, controlled counterfactuals attribute a measured fraction of the edit–quantization interaction to a specified component of edit-induced rounding,” or a bounded failure of that explanation. No general law or universal safety proxy follows.

## 2. Cheap fallback: projected-estimator conditioning

**Does harmless-mean projection amplify small, explicitly controlled centroid perturbations into larger held-out behavioral changes than comparable difference-vector perturbations?** This studies numerical conditioning of the recovered estimator, rather than inventing a new removal method. It is a sensitivity analysis, **not a bootstrap estimate of sampling uncertainty**.

The actual `refusal_directions.pt` archive stores `good_means`, `bad_means`, `directions`, `raw_directions` and a flag; the corrected notebook saves these reduced tensors, **not per-prompt activations**. No separate per-prompt activation cache was found in the local artifact inventory. Consequently these means cannot support prompt bootstrap or confidence intervals for the direction estimator.

Write `r(d,g)=normalize(d−projection_g(d))`. From saved means alone, compute `κ=||d||/||d−projection_g(d)||` and finite-difference angular sensitivity using a preregistered 1% relative-norm perturbation and fixed random seeds. Small projected residuals can amplify estimation error. Freeze `d` while perturbing `g`, then freeze `g` while perturbing `d`; choose representative vectors using geometry only. On one frozen FP16 original compare four directions: reference projected, centroid-perturbed projected, difference-perturbed projected and angle-matched random perturbation. Hold unit norm, strength, layer and application site fixed; verify a zero-intervention sham separately. Use 24 independent prompts, balanced across harmful, sensitive benign and utility tasks, for **96 target-model completions or 1 quota-hour**, including setup, loading, classification/scoring and any capture. Do not rerun the optimizer.

A positive result is disproportionate behavioral sensitivity to the controlled centroid perturbation; unchanged geometry/behavior is an informative null. Report refusal, assistance, over-refusal and utility separately. These synthetic perturbations need not resemble real sampling error, so the result cannot establish finite-sample instability. A genuine bootstrap extension requires **new per-prompt activation capture on Kaggle**, disjoint from evaluation. Include that capture in the same one-hour cap or defer it; no runtime is known. A cross-precision extension likewise needs new capture and the factorial `r(d_FP16,g_FP16)`, `r(d_Q,g_FP16)`, `r(d_FP16,g_Q)`, `r(d_Q,g_Q)` on one unchanged quantized original, with an unedited control. Neither extension belongs to the mean-only cheap pilot.

**Legitimate claim:** local conditioning and behavioral sensitivity of this saved projected estimator under specified synthetic perturbations. Bootstrap consensus and quantized extraction already exist in practitioner tooling; do not claim invention of either. This behavioral conditioning audit appears less directly covered in the sources searched, but novelty confidence is lower than for a successful causal decomposition.

## Shared measurement and limits

Freeze model/dataset revisions, prompt IDs, template/thinking setting, generation settings, target matrices, vector files and classifier version. Generate to EOS with the same cap; separately report truncation and extend a predefined capped subset in every compared arm. Our 100-token marker result and heavily capped 512-token examples cannot substitute for semantic evaluation.

Score **semantic refusal, substantive harmful assistance, benign over-refusal, task correctness, and incoherence separately**. Use a fixed safety classifier plus blinded human adjudication of disagreements and a stratified sample; report denominators, paired uncertainty and label ambiguity. Benign KL is a collateral-change diagnostic, not a utility or safety guarantee. A full held-out test should follow the pilot only if it exposes a measurable, interpretable effect; do not select vectors on that test.

Qwen3.5-9B's DeltaNet/full-attention/MLP mixture makes module accounting essential. Audit where each edited Linear writes relative to gates and normalization; algebraic weight corrections must occur at that exact Linear, not an assumed universal residual hook. If the primary effect exists, module-group corrections can localize it, with equalized correction norms as controls. This is a within-model contrast: layer depth and module type are confounded, and [prior hybrid steering work](https://arxiv.org/abs/2609.04721) prevents claiming that hybridity itself is new. Two T4s and the remaining 25.37 quota-hours make one bounded pilot plausible; no measured runtime yet supports a larger matrix.

## Primary-source literature audit

Dates below are source-listed submission/publication dates; preprints are identified as such. The table records coverage, not endorsement of every result.

| Primary source and verified date | Already covered | Remaining distinction relevant here |
| --- | --- | --- |
| [Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717), submitted 17 Jun 2024; NeurIPS 2024 | Causal removal/addition of refusal directions; weight editing and selection controls. | Established foundation; not an edit-induced quantization-error attribution study. |
| [Chhabra & Khalili, *Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability*](https://arxiv.org/abs/2504.04215), 5 Apr 2025, preprint | Quantized/pruned refusal mechanisms; high FP/quantized direction similarity; AIRD safety repair evaluated for pruned models. | Rules out claiming the compression/refusal intersection is unexplored. |
| [Chen et al., *Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models*](https://proceedings.mlr.press/v267/chen25ci.html), ICML 13–19 Jul 2025 | Safety across quantizers/calibration data; Q-resafe patching. | Generic safety degradation and repair already covered; our estimand concerns a fixed pre-existing edit. |
| [Egashira et al., *Mind the Gap: A Practical Attack on GGUF Quantization*](https://proceedings.mlr.press/v267/egashira25a.html), ICML 13–19 Jul 2025 | Quantization-error-constrained malicious weight changes; behavioral differences hidden by precision. | Rounding can carry behavior already known; not evidence for our specific mechanism. |
| [Wee et al., *Safety-Preserving PTQ via Contrastive Alignment Loss*](https://arxiv.org/abs/2511.07842), submitted 11 Nov 2025; revised 18 Mar 2026 | Safety-aware W4A4 calibration using aligned/unaligned references. | Calibration itself is an intervention; hold it fixed before attributing rounding effects. |
| [Kadadekar, *Quality Is Not a Safety Proxy Under Quantization*](https://arxiv.org/abs/2606.10154), 8 Jun 2026, preprint | Broad matched safety/quality audit; direction geometry and other mechanistic proxies weak or null separators. | Mere cosine drift is inadequate; proposed correction is causal and edit-specific. |
| [Bosco & Srinivasan, *Locating and Steering Refusal Beyond Attention*](https://arxiv.org/abs/2609.04721), 4 Sep 2026, preprint | SSM/recurrent/hybrid refusal; write-site estimation and causal steering transfer. | Hybrid refusal is not new; numerical survival of this compiled edit remains a scoped question. |
| [He et al., *Disentangling Representation Evolution in Transformers through Directional Decomposition*](https://arxiv.org/abs/2609.15975), 14 Sep 2026; lists Findings EMNLP 2026 | Parallel/perpendicular update decomposition and compression-error diagnosis. | Decomposition itself is not new; ours is relative to a frozen refusal edit, with behavioral corrections. |
| [Prosvirnin et al., *Amplifying Latent Safety in Large Language Models via Gated Activation Steering*](https://www.mdpi.com/2079-3197/14/10/261), 21 Oct 2026 | Gated harm-recognition/refusal steering evaluated in BF16/NF4, safety and utility. | Neither quantized steering nor its utility trade-off is a new contribution. |
| [Ko, *Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan*](https://arxiv.org/abs/2610.29342), 21 Oct 2026; v2 22 Oct, preprint | Matched quantized original/edited models; distinguishes refusal from stance; cites partial NVFP4 edit survival in Mamba matrices. | Edit-survival anecdotes already published; our proposed addition is controlled causal attribution. |

An additional OpenReview search surfaced *The Channel Geometry of Refusal: Mechanistic Diagnosis of Alignment Collapse Under KV Quantization* ([primary PDF](https://openreview.net/pdf?id=n1CfsXNCSL)). The primary page/PDF was blocked by verification, so I could not verify its exact publication date or full protocol and did not treat snippets as established evidence. Check it before any priority claim; KV/channel error mechanisms could narrow the gap further.
