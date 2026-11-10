# Report triage and implemented revision

Revision: 10 November 2026. Owner: independent report triage-and-fix agent. This is an author-side record, not report content.

## Baseline preserved

The original 28-page draft, its two Python sources, bibliography plan, claim map and key QA were copied to `baseline_v1_20261110/` before authoring. Baseline DOCX SHA256: `c6b4eedb1bf6287e05f84d2caddb573ef631470005c1930437aeb88dafb6dc49`. The supplied `SampleReport.docx` is untouched.

## Triage and changes actually made

| Finding in v1 | Implemented fix |
|---|---|
| Abstract opens with general methodological claims but has no supporting citations. | Rewrite the abstract around the actual bounded study; cite Arditi for linear refusal intervention and StrongREJECT for the evaluation distinction. Do not use those citations to validate our measured counts. |
| Related work is mainly a sequence of brief source summaries and implementation caveats. | Connect refusal control, category geometry, semantic evaluation and compression safety. Add Table 2.1 comparing each study's contribution with its implication for our question. Explain the inference from literature to the fixed-edit comparison. |
| Objectives are dominated by reconstructing and debugging the run. | Distinguish immediate RQ1–RQ3 from a future prospective diagnostic question. State a focused review plus checked exploratory baseline as the contribution. No exhaustive survey, new method or established publication novelty is claimed. |
| Hash errors, packaging exceptions, resource accounting and repeated audit descriptions crowd the main argument. | Remove operational details from chapters; retain exact evidence routes in the separate claim map. Reduce to one appendix page of essential public revisions, fixed schedule and runtime. |
| Methodology and limitations repeat the same disclaimers. | Combine fixture/settings/rubric descriptions, keep key limits next to numerical results and a compact threats paragraph. Preserve the distinctions that change interpretation. |
| The study diagram mainly documents workflow. | Replace it with a plain study-logic figure connecting literature themes, the fixed-edit question and actual comparison controls. No effect plot or unmeasured mechanism is invented. |
| Report could conflate NF4 feasibility with validated stability or compare independent label sets silently. | Retain the pilot's original labels, separate them from the later audit, define the interaction, and state that the finite-fixture bounds are not confidence intervals. Explain refitted scales and frozen factors. |

The revised title is **Refusal Direction Ablation and Quantization Stability**, subtitled **Literature Review and an Exploratory Qwen Model Study**. It retains the college template's single-column Times New Roman layout and front matter. It does not imitate an IEEE two-column proceedings layout. Chapter 2 now occupies four pages; methodology and provenance are shorter. The complete report is 26 pages and about 5,159 rendered words, including front matter and references.

## Frozen literature and grounded use

The twelve source identities and reference metadata are unchanged from v1. `BIBLIOGRAPHY_PLAN.md` is marked frozen; final citation numbers follow first appearance, including the abstract. No source proved invalid, so none was replaced or added.

| Final citation | Primary material inspected | Supported use and boundary |
|---|---|---|
| [1] Arditi et al. | [Paper, methods and Appendix C](https://arxiv.org/html/2406.11717v3) | Activation mean differences, direction removal/addition, validation-based selection with refusal induction and harmless KL constraints. Evidence is for studied chat models, not a universal refusal mechanism or our model's accuracy. |
| [2] Souly et al. | [StrongREJECT methods and evaluator validation](https://arxiv.org/html/2402.10260v2) | Useful forbidden information and comparison with human judgments; motivates separating useful fulfillment from non-refusal. We did not run its calibrated evaluator. |
| [3] Dettmers et al. | [QLoRA method](https://arxiv.org/html/2305.14314) | NF4, quantized weights with floating computation, low-rank adapter training, double quantization and paged optimizers. Our experiment uses NF4 packing only. |
| [4] Chen et al. | [ICML/PMLR primary paper](https://proceedings.mlr.press/v267/chen25ci.html) | Safety assessment across quantization/calibration and quantization-aware safety patching. Establishes existing prior work; does not validate our frozen-edit diagnostic or NF4 outcome. |
| [5] Heretic | [Pinned implementation README](https://raw.githubusercontent.com/p-e-w/heretic/e26da5e0e648bb285f896c9801849edfa8d20913/README.md) and locally audited pinned source | Refusal/KL optimization and directional editing. Our fixed-recipe reconstruction is not a fresh optimization study. Projection/formula provenance also rests on audited code, not only the README. |
| [6] Qwen | [Official model card](https://huggingface.co/Qwen/Qwen3.5-9B) and pinned config | Model identity and hybrid Gated DeltaNet/full-attention architecture. Not evidence of our benchmark performance. |
| [7] Abliterlitics | [Project comparison](https://abliterlitics.dev/models/qwen3.5-9b/) and [pinned classifier](https://github.com/dreamfast/abliterlitics/blob/c48b882660daffd06f1c01ba59f1e3d114f768f8/src/benchmark/harmbench_classify.py) | Source-reported 0/400 and heuristic implementation. Kept distinct from official HarmBench and our own protocol. Historical exact settings/raw answers are unavailable; cause of the discrepancy is unresolved. |
| [8] Mazeika et al. | [HarmBench primary publication](https://proceedings.mlr.press/v235/mazeika24a.html) | Standardized behaviors and evaluation framework. Selected safety cases are not a full official benchmark run. |
| [9] Röttger et al. | [XSTest design and findings](https://arxiv.org/html/2308.01263v3) | Sensitive safe prompts and unsafe contrasts; motivates over-refusal coverage. Only five distinct safe items were executed here. |
| [10] Vaswani et al. | [Original Transformer paper](https://arxiv.org/abs/1706.03762) | Scaled dot-product attention, heads, causal masking and residual connections. Qwen's hybrid architecture is separately sourced. |
| [11] Joad et al. | [EMNLP 2026 primary record](https://aclanthology.org/2026.emnlp-main.2606/) and [full methods/findings](https://arxiv.org/html/2602.02132v2) | Category-dependent directions, shared steering trade-offs and shared/category-specific features in studied models. Does not explain the mechanism of our observed redirects. Correctly identified as EMNLP 2026. |
| [12] Kadadekar | [2026 preprint methods, matrix and limitations](https://arxiv.org/html/2606.10154v1) | Quality can coexist with reduced refusal in tested conditions; weak geometric separation and within-matrix screening motivate prospective validation. Explicitly a preprint, not a verified conference publication or transferred diagnostic. |

Connecting these findings to a fixed-edit stability question is our synthesis, not an externally proven mechanism. The report explains that quantization can change codes and scales, but does not infer a behavioral failure from algebra alone. It makes no first-ever claim about the broader question.

## Results retained and interpretation boundaries

The historical 100/72/8/8/8 marker counts, cap counts and first-token KL are unchanged. The 64 factor pairs, 42 active matrices and 100/100 save/reload matches remain. Table 4.2 retains matched and fresh semantic counts for original, reconstruction and publisher. Table 4.3 retains the original pilot labels and Table 4.4 the independent metric disagreement counts. Exact internal evidence is in `CLAIM_TRACEABILITY.md`.

The reconstruction is checked but not byte-identical to the publisher. Identical categories across fourteen selected safety cases are not a statistical equivalence result. One automated reviewer, imperfect masking, unverified harmful accuracy and substantial truncation remain explicit. Arithmetic and safe subsets are elementary checks; no broad capability-preservation claim is added. The NF4 result remains feasibility with unresolved interaction, not a validated failure, predictor or repair.

## Verification

Canonical bundled rendering produces 26 pages. Every page was individually inspected at readable resolution. Tables, formulae, figure labels, references, footers and contents/list locators were checked. A fresh template-diff render matches all 26 inspected page images exactly. There is no clipping, overflowing page content, split table row, orphan caption or unexplained blank page.

The package preserves 13 original parts and all original relationships. Only the five allowed existing XML parts and the added figure change. The original sample SHA is unchanged. The report has 18 sections, 8 tables, 1 figure, 4 native displayed equations, 12 cited references and zero unresolved citation tokens, tracked changes or comments. Full Word-text/relationship scanning finds no internal research artifact route, private checkpoint ID or unresolved source token. QA details are in `writing_qa/structural_checks.json`, `self_containment_scan.json`, `build_manifest.json` and `QA_NOTES.md`.

Student, ID, programme, supervisor, department, institute, emblem, examination date and signatures remain editable placeholders. No institutional approval or human review is invented.
