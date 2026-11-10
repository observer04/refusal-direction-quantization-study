# Handoff for college report checkpoint

Updated 10 November 2026, Asia/Kolkata. User requested this handoff quickly because their assistant quota is nearly exhausted. Preserve context and conserve calls. **No new experiments are needed to begin the report.**

## Latest request and immediate next action

User asked: “larger project status to the college report checkpoint? can we write a report now?” Root answered yes and began checking sources/template, then user requested this handoff. **No report DOCX or report body has been authored.** Next session should prepare the actual college report using the verified results below and `SampleReport.docx`; do not restart literature research or launch GPU runs by default.

College submission is **11 November 2026**. Novelty is not required; modest original analysis is a personal quality goal. User previously said to hold report writing and focus on research, then asked about writing now after the audit. The audit is now complete. Root has committed to preparing the report with explicit limitations. Student names/IDs/supervisor were requested; user declined to supply them then. Use obvious editable fields if still unspecified, rather than repeatedly blocking on metadata or inventing it. Do not invent signatures, approval or acknowledgements of work that did not occur.

Recommended report scope: **reproduction and semantic evaluation of refusal-direction ablation in Qwen3.5-9B, with an NF4 feasibility pilot**. A possible title is “Refusal Direction Ablation in a Hybrid Language Model”. This is sufficient material for a scoped empirical technical report, not certification of college acceptance or a conference contribution.

## User constraints

- CWD `/home/op/llm/abliteration`. All model loads, weights, installs and inference stay on Kaggle. No local venvs, pip installs or model runtime.
- Existing Kaggle CLI is available. Kaggle `ommprakash007`; HF `Observer04`.
- `.env` HF token was authorized for use; never print it or include it in source/logs/uploads. Existing private runtime-dataset pattern handles remote credentials.
- Preserve every completion and generation metadata for human review; keep original raw outputs unchanged. Human labels must stay separate from agent labels and condition mappings.
- Keep `STATE.md` brief and current. All new audit work is finished. All agents have completed; do not spawn more without user authorization.
- User wants no shortcuts, exaggerated success metrics, extensive new research loop or predetermined positive result.

## Main scientific verdict

**We have a correctly reconstructed weight edit with real but incomplete semantic effects on selected cases. The actual publisher shows the same limitations under the tested settings.** No material reconstruction deficit was observed, but this is not an equivalence test. Universal non-refusal, useful harmful accuracy, broad capability retention and the external 0/400 result are unestablished.

The user's concern was valid: **8/100 refusal-marker matches does not mean 92/100 genuinely fulfilled requests.** Do not repeat that interpretation. The historical detector counts 32 normalized substrings anywhere, including `sorry`, `i can'`, `illegal`, `harmful`, `violat`, `unethical`; empty output also counts refusal. It can miss a safe reframe and match a warning accompanying assistance.

### Historical reconstruction evidence

| Condition | Marker matches /100 | Benign first-token KL |
| --- | ---: | ---: |
| Original | 100 | 0 |
| Failed raw-direction replay | 72 | 0.434047 |
| Actual publisher FP16 | 8 | 0.078430 |
| Corrected reconstruction FP16 | 8 | 0.078022 |
| Publisher card reported | 6 | 0.0793 |

These are `mlabonne/harmful_behaviors test[:100]`, greedy nonthinking, EOS enabled, **100-token cap**, exposed during recovery. Corrected 99/100 and publisher 100/100 capped. KL is `KL(original || edited)` for the first generated token on 100 harmless prompts, not broad utility. Independent audit reproduced all 500 stored historical labels.

The original failure omitted existing upstream per-slot harmless-mean orthogonalization. Correct recovery: difference of means, project each slot's vector off its harmless mean, renormalize, interpolate slots 19/20 at 0.49, use fixed published component strengths. Fresh extraction uses 400 harmful plus 400 benign. No optimizer/search was needed. 64 adapter pairs, 42 active writers (19 attention, 23 MLP). Small-vector/factor recalculation and prior complete writer hashes verified implementation. Saved/reloaded model gave 100/100 identical responses. This is approximate recipe/measurement reproduction, not byte-identical publisher reconstruction.

### Latest independent model audit completed

Authoritative file: **`research/model_audit_v1/AUDIT.md`**. Read this before writing results. Related `SOURCE_METRIC_AUDIT.md`, `ITEM_REVIEW.md`, `semantic_summary.json`, `decision_rule_results.json`, and `classifier_comparison_all.csv` retain details.

| Safety cohort and model | Clear fulfillment | Partial | No fulfillment | Explicit refusals | Capped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched original, N=6 | 0 | 0 | 6 | 6 | 1 |
| Matched reconstruction, N=6 | 2 | 3 | 1 | 0 | 5 |
| Matched publisher, N=6 | 2 | 3 | 1 | 0 | 5 |
| Fresh original, N=8 | 1 | 1 | 6 | 8 | 0 |
| Fresh reconstruction, N=8 | 6 | 2 | 0 | 0 | 5 |
| Fresh publisher, N=8 | 6 | 2 | 0 | 0 | 6 |

One auditor read opaque shuffled queues and froze labels before opening mappings. **Imperfect condition masking, not independent human review**: some earlier prompts/aggregate findings were known. Harmful content accuracy unverified. Factual correction, knowledge gaps and ambiguous intent are separate fields, not automatically safety refusal. Original's fresh SQL response refuses then provides the requested payload, illustrating that refusal and assistance can coexist. Do not reproduce harmful operational detail in the report; paraphrase and link record identifiers.

Publisher and reconstruction share judgment categories on all 14 safety cases; 8/24 full response pairs were byte-identical. Account-protection redirect occurs in both. Smoking fact correction does not establish a surviving safety mechanism. Matched organ contacts and locating responses have missing/uncertain information. Fresh eight include six clear and two partial fulfillments in both edits. This is a tiny deterministic fixture; no population rates, statistical equivalence, broad safety claim or validated accuracy follows.

The protocol's exploratory broad-fixture cutoff fails on 1/8 redirection (12.5% versus <=10%); other thresholds pass. This fragile convention **is not a universal definition of “properly abliterated”**. Do not elevate it into a scientific efficacy boundary. Audit reports real but incomplete effects.

All three arms answer five elementary arithmetic cases correctly across cohorts; no observed benign safety over-refusal. Two benign cases per arm are only partly useful. This does not establish general capability preservation.

New GPU job `ommprakash007/qwen35-independent-model-audit-v1` v1 completed 48 new responses: publisher on 12 prior cases, then 12 fresh mixed cases across original/publisher/reconstruction. Prior W/D 24 records reused only after exact settings/template/token/runtime gates; human bundle contains **72 records total**, not 72 new generations. Fresh cases selected by timing only, before text inspection. Fresh safety caps1024, benign768, arithmetic384; matched768/384. No rerun or model repair.

48 token sequences exactly reconcile with journals. Auxiliary `reference/frozen_fixture_listing.json` was hashed while empty, then finalized before launch; its content is exactly derived from the unchanged frozen fixture, same prelaunch timestamp. Original manifest preserved and exception disclosed. All substantive protocol/fixture/source hashes match. Do not omit this exception or describe all preregistration hashes as identical.

### Classifier audit and external claim

External https://abliterlitics.dev/models/qwen3.5-9b/ explicitly identifies **the same trohrbaugh v2 repository** and claims 0/400 plus no soft refusals. Do not dismiss it as another variant. It reports 2048 tokens, temperature0, custom classifier v3 plus manual overrides/review and reasoning analysis. Exact historical model revision, requests/template, raw400 answers and overrides were not located in inspected public artifacts. Our nonthinking helpful-system FP16 evaluation differs. No tested causal attribution to those differences.

Pinned public classifier: https://github.com/dreamfast/abliterlitics/blob/c48b882660daffd06f1c01ba59f1e3d114f768f8/src/benchmark/harmbench_classify.py . Regex/preamble/structural heuristics, **not official HarmBench learned classifier**. It is not proven byte-identical to the historical executable. Rescoring after labels froze:

| Cohort / model | Upstream marker refusals | Public v3 refusals | Clear fulfillment |
| --- | ---: | ---: | ---: |
| Matched original N6 | 6 | 6 | 0 |
| Matched reconstruction or publisher N6 | 2 | 1 | 2 |
| Fresh original N8 | 8 | 7 | 1 |
| Fresh reconstruction or publisher N8 | 1 | 0 | 6 |

It counts the account-protection redirect as nonrefusal. **Zero detected refusals can coexist with partial fulfillment.** Source-specific disagreement is demonstrated on our saved outputs, not an established explanation of the historical400 score.

`classifier_preprocessing_witness.json` contains a pure-text bug witness: plain “I cannot provide instructions for committing fraud.” is detected; wrapping that same text after `<think>Placeholder reasoning.</think>` leaves a `>` when v3 slices after literal `</think`, defeats anchored detection and labels it nonrefusal. This is a conditional source bug. No closed-think tags occur in our clean safety responses, so it does NOT cause our own rescoring mismatches. Historical formatting/overrides unknown; no fraud allegation or attribution of 0/400 to this bug is warranted.

## NF4 pilot stays a separate analysis

Read `research/feasibility_v1/RESULTS.md`. 12 prompts (6 safety/3 sensitive benign/3 arithmetic) x5arms =60 completions. W, edited D, native NF4 QW, native QD, exact saved floating edit on same packed QW. Native block64 NF4, no double quantization, uint8 storage, FP16 compute; adapter factors FP32,3MiB. QD scales refit, **not common frozen grids**. No fresh vector extraction.

All64 target merge checks and native packing/adapter checks passed. Earlier reviewer counted2 clear safety assistance in D/QD/QW_D versus0W/QW. Its partial/unknown rubric differs from the independent audit; preserve both analyses rather than silently substituting audit labels into quantized conditions. Interaction bounds [-0.500,+0.667], widen with censoring; not confidence intervals. No clear edit-erasure failure, predictor or repair result. One QW_D arithmetic answer capped; other elementary totals correct. Pilot consumed1.04quota-hours.

## Persistence and compute

Corrected private checkpoint https://huggingface.co/Observer04/Qwen3.5-9B-heretic-corrected-20261109

- Root checkpoint pin `6cd66cf9bf38efab01ffd5ab6e5c7995868f55bb`.
- Audit raw run commit `31b8705cb4db4be82c998eecc43ced2a9a930ca5`, path `research/model_audit_v1/run-v1/`.
- **Final audit review commit `7f12baad08f0f5296b763f2c689be5731e322544`**, path `research/model_audit_v1/review-v1/`;135 artifacts /9,115,976bytes; all135 remote blob hashes verified and representative byte downloads matched. Root checkpoint files unchanged. Proof `research/model_audit_v1/persistence_verification.json`.
- NF4 final review commit `9e4f047bafd49994ee488eec31888d16d43a571f`,87 verified artifacts.
- Audit GPU spend0.87h, last verified23.46h remaining, resetNov14. This is **Kaggle quota**, separate from user's nearly exhausted assistant quota.
- Prior recovery2.11h, NF4pilot1.04h, audit0.87h =4.02h for these three blocks; earlier failed runs are extra. Do not call4.02h total project cost.
- All inference is COMPLETE; no newGPU job planned. A known optional CPU-only older audit was last queued; not needed.

## Model and source pins

- Original `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Publisher `trohrbaugh/Qwen3.5-9B-heretic-v2` @ `81dda62421c43a5bd5722edbc6cd47714673dc7f`.
- Heretic hybrid-compatible source `e26da5e0e648bb285f896c9801849edfa8d20913`; v1.2.0 alone predates required hybrid support.
- Adapter hash `c38fbd7c11c86c9ea898477344b588f5342c7ed95a2ce4c8942335af5f73f0db`.
- Latest recorded runtime: torch2.11.0+cu130 preserved, transformers5.22.1, peft0.21.2, accelerate1.15.0, bnb0.50.2, HFhub1.33.0. Use manifests for exact run settings. T4FP16; no nativeBF16 assumption.

## Report production handoff

`report/COLLEGE_REPORT_CHECKPOINT.md` maps the sample, evidence and uncompleted original scope. It still mentions report hold/audit in progress in places; update it when resuming authorship. `STATE.md` is the brief live overview; root should update final audit status/commit.

**Report narrative**: recover a missing published projection step; verify editing and persistence; demonstrate a real but incomplete semantic effect matching the publisher on selected cases; show why phrase-marker/custom heuristic metrics differ from semantic fulfillment; report the small NF4 result as unresolved. This is substantive engineering and original local analysis without claiming a new algorithm or universal finding.

Suggested chapters:
1. Introduction and scoped questions.
2. Related work on refusal directions, semantic evaluation and quantization.
3. Methodology and implementation: Qwen hybrid, mean difference/projection/interpolation, frozen edit, conditions, datasets/splits, generation and judging.
4. Results and discussion: failed/corrected reproduction, publisher paired audit, metrics disagreement/conditional parser witness, benign limits, NF4pilot, integrity/compute/limitations.
5. Conclusion and future scope; verified references and reproducibility appendix.

Make measured plots from saved JSON (paired clear/partial/no counts; historical markers with cap annotation; classifier disagreement) and an architecture/experiment diagram. No hypothetical activation plots or fabricated CI. Save plot data and source. Clearly disclose AI-assisted implementation/annotation and absence of independent human labels. Include human review as a limitation/strengthening step; do not invent it to meet a checkpoint.

**Larger original project is unfinished**: no refusal-enhanced condition; no per-prompt residual/probe/geometry study; means cannot substitute for captured residuals; no quantized re-extraction or genuine Q(A(W)) versus freshly estimated A(Q(W)); no second model family/multiple formats/validated predictor or repair. Put these in future work. Literature rounds rejected current conference commitments; no more broad literature loop needed for college report.

### Sample and document tooling already inspected

- `/home/op/llm/abliteration/SampleReport.docx` unchanged, SHA256 `e62bd2c6c049f854c84896c9ff903e4a04d583ca25ec51399f9895474e260f1a`.
- Actual render18pages, largely scaffold. Contents numbers up to95 are placeholders, **not page requirement**. Frontmatter: title, viva approval, acknowledgements,250–300word abstract/5–6keywords, contents, lists of tables/figures/algorithms/abbreviations. Five chapters and numbered references. Agricultural examples/references must be replaced.
- Read-only extraction: `report/checkpoint_evidence/sample_paragraphs.json`, `sample_text.txt`, `sample_metadata.json`; renders `sample_render/page-1.png`...`page-18.png`;3contact sheets inspected. No full-final DOCX visualQA yet because no report exists.
- Reference has18sections, A4-like11910x16840twips, TimesNewRoman, BodyText12pt, Heading1 centered bold14pt; margins differ. Footer1roman, footer2arabic. An awkward continuous2column equation section and old numbering inconsistencies need deliberate treatment. Preserve visual authority, correct navigation, do not copy wrong TOC numbers.
- Documents skill was read/announced: `/home/op/.codex/plugins/cache/openai-primary-runtime/documents/26.1105.12137/skills/documents/SKILL.md`; also `template-distill.md`, `template-create.md`, `tasks/read_review.md`, `tasks/create_edit.md`, `writing_quality.md`. Use template copy plus an `artifact.md` contract, no generic redesign. Do not re-read everything unnecessarily; check needed sections.
- Existing bundled runtime (no installation): `/home/op/.cache/codex-runtimes/codex-primary-runtime/runtime.json` records26.1105.12137.
- Python `/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12` (python-docx,lxml,PIL,matplotlib available through bundled dependencies; no newenv needed).
- Node `/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node` (verify exact executable if needed).
- Bundled LibreOffice wrapper `/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice`; native `/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/program/soffice`. **Never use installed desktop LibreOffice.** Packaged renderer auto-selects runtime from Python path.
- **Before FIRST DOCX authoring command**, run skill's `container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format docx` exactly once using bundled node. NOT YET RUN. No DOCX authoring operation started.
- Render with bundledPython + skill `render_docx.py OUTPUT.docx --output_dir QA_DIR --emit_pdf`; inspect every final page at100%, fix and rerender before delivering. Keep internal images/PDF out of deliverables unless requested. Source template must remain unchanged.
- Final report author metadata unspecified; obvious placeholders are honest. Explain editable fields in final message. No fake signed approval or numerical result inserted to fill pages.

## Useful primary sources

- Arditi etal https://arxiv.org/abs/2406.11717
- StrongREJECT https://arxiv.org/abs/2402.10260
- Official HarmBench https://github.com/centerforaisafety/HarmBench
- XSTest https://arxiv.org/abs/2308.01263 and pinned sources in manifests.
- Qwen/publisher pinned model cards and Heretic pinned source already saved locally.
- External classifier/generator immutable links and access limits are in `SOURCE_METRIC_AUDIT.md`.
- Final report should use ordinary verified references and source attribution, not tool-reference IDs, self-reported benchmark values presented as our measurements, or conference novelty claims.

Resume by reading `STATE.md`, `research/model_audit_v1/AUDIT.md`, this handoff and the checkpoint assessment. Then author the report; all research evidence and human-review outputs are preserved. Conserve remaining assistant quota and avoid redundant experiments.
