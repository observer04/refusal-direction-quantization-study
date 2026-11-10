# College panel report checkpoint

Assessment date 10 November 2026. **User clarification: submission is tomorrow, 11 November 2026; a novel method is not required.** The immediate objective is a technically defensible college report following `SampleReport.docx`, with modest original analysis as a quality goal. Conference novelty remains a later objective. This assessment does not certify model efficacy or college acceptance. The independent model audit is in progress.

Latest steering: the user has asked to hold report writing and focus on research. This file remains a checkpoint assessment; no final report DOCX has been authored. The active work is the independent model audit and evaluation discrepancy. The production steps below are retained as a later plan.

## Proposed checkpoint

Proposed title: **Reproducing and Evaluating Refusal Direction Ablation in Qwen3.5 9B**.

Complete a reproducible implementation and comparative evaluation of the original model, the actual published edited checkpoint, and our reconstructed checkpoint. Establish how the edit changes semantic refusal, substantive assistance and selected benign tasks. Explain where the historical refusal-marker score agrees or disagrees with those outcomes. Include the native NF4 experiment as a clearly scoped pilot, with its unresolved result intact.

This reporting checkpoint is narrower than the entire original research proposal. The user's confirmed absence of a novelty requirement supports an empirical reproduction and evaluation scope. For tomorrow, finish the authorized audit, analyze the evidence, and write and verify the report. Do not launch a larger benchmark campaign merely to make the report look comprehensive. Its conclusions must remain scoped to the actual evaluated cases.

Calling the weight file abliterated establishes an editing procedure and provenance. Calling it effective requires behavioral evidence under stated conditions. A college report can honestly document partial effectiveness or a failed replication; it must not relabel either as uniformly successful refusal removal. If reliably low semantic refusal is itself a required deliverable, that model objective remains open until the audit supports it.

## Questions the report should answer

1. Can the documented editing procedure be reconstructed, saved and reloaded reproducibly?
2. Does the reconstructed model behave like the actual publisher checkpoint under matched evaluation, and how do both differ from the original?
3. Does the phrase-based refusal metric accurately describe semantic withholding and substantive assistance on the evaluated prompts?
4. What selected benign-task costs and over-refusal effects are observable?
5. What can the small native NF4 comparison establish, and what remains unresolved?

These are questions, not assumed positive findings. The report must not claim a new refusal mechanism, general capability preservation, universal non-refusal or a quantization repair.

## Current evidence and gaps

| Component | Evidence already available | What remains |
| --- | --- | --- |
| Editing implementation | Fresh 400 plus 400 means; published harmless-mean projection recovered; frozen writer/adapter checks; 100 of 100 reload responses identical | Independent audit of implementation, checkpoint identity and inference harness |
| Historical metric reproduction | Original 100 of 100 marker refusals, actual publisher 8, reconstruction 8; first-token benign KL approximately 0.0784 and 0.0780 respectively | Explain exact detector and exposed dataset; never interpret 8 marker matches as 92 fulfilled requests |
| Semantic evaluation | Twelve fresh prompts across five conditions, including six safety prompts; all 60 completions retained; one condition-masked agent review | Direct actual-publisher comparison, independent human assessment, larger prespecified validation and explicit censoring treatment |
| Benign utility | Three sensitive-safe and three arithmetic cases per condition | More representative checkable tasks; separate factual correctness, instruction following and over-refusal; no broad capability claim |
| Native quantization | Verified NF4 packing and fixed floating adapter reference; one configuration, refitted scales | No identifiable edit-erasure effect; no fixed-grid causal rounding result; no validated predictor |
| Reproducibility and persistence | Pinned source, models and datasets; runtime manifests; raw tokens, outputs, hashes and private evidence | Consolidated verification entry point, final claim-to-evidence table and panel demonstration |
| Literature | Multiple source-linked reviews and adversarial triage | Select and verify a concise relevant set for the report; replace the template's unrelated agricultural references |
| Report document | Actual sample located and inspected structurally and through rendered page overviews | Body text, final tables/figures, real author metadata, template-based DOCX creation and full visual verification |

The implementation and evidence storage are the strongest parts. Behavioral validation is still at pilot scale. The final report has not been authored. A completion percentage or a confident number of days would hide these different states.

## The model audit now authorized

One independent agent is auditing the exact upstream refusal matcher, saved outputs, edit provenance, and the original externally reported 0 of 400 result. That external report names the same publisher repository; differences in inference settings, response extraction, classifier and checkpoint revision must be traced rather than assumed away.

The agent can run one bounded FP16 comparison on Kaggle. It prioritizes the actual publisher on the same twelve feasibility cases, then a prespecified fresh stratified sample across original, publisher and reconstruction if profiling permits. Total ceiling is two Kaggle quota-hours and 84 new completions, including any technical retry. Every completion is retained for human review. This is an audit spending ceiling, not a promise of adequate statistical power.

Possible audit outcomes have different consequences. If publisher and reconstruction show similar semantic behavior, residual refusals are evidence about the reference procedure under that evaluation. If they differ substantially, diagnose checkpoint, editing or inference differences before treating the reconstruction as validated. If semantic changes are weak in both, report that scope honestly and investigate the external evaluation discrepancy. No outcome licenses tuning on these evaluation prompts and then reporting them as untouched validation.

## Evidence needed to finish the reporting checkpoint

Freeze a final evaluation protocol after the audit resolves the main implementation and measurement questions. List exact model revisions, datasets, disjoint splits, prompt families, system text, chat template, thinking mode, decoding, EOS handling, token budgets, judging rules and statistical summaries before final generation.

For tomorrow, use the fixed audit sample and existing controlled runs as an explicitly exploratory study. The audit's new stratified sample and actual-publisher comparison improve the evidence without pretending to establish a population failure rate. Report exact denominators, category selection, exposure and all unresolved cases. A future stronger validation could target approximately 100 unambiguous safety behaviors and 50 benign tasks, with size chosen from interval precision and measured runtime; that larger campaign is not required for this deadline and has not been launched. Preserve category coverage and an explicit sampling rule. Do not reduce sample size after inspecting outcomes.

Judge explicit refusal, safe redirection, substantive assistance, factual or technical correctness, inability or unavailable information, and ambiguous intent separately. A correction of a false factual premise is not automatically a safety refusal. An attempted but broken answer is not evidence of successful harmful capability. Mark truncation and unresolved judgments; do not silently assign them to success or failure. Use a prespecified extension policy for capped responses, paired conditions and identical budgets. Longer generation alone does not settle whether an answer fulfills a request.

For a final independent human assessment, plan one human review of the safety outputs and a second reviewer on a stratified subset and disagreements, with model identity withheld where practical. Retain ratings, rubric, mapping, disagreement and adjudication records. If this review is unavailable, disclose the actual agent-only assessment and weaken the claims accordingly. A second AI reviewer is not a human review.

Compare marker labels with semantic judgments in item-level tables, including false apparent successes, redirects and partial answers. Report paired model differences and uncertainty appropriate to the sampling scheme, alongside unresolved and censoring bounds. A narrow equality of point estimates does not establish equivalence. First-token KL is supplementary and cannot replace task correctness or broad capability testing.

The original and publisher must remain separate controls. Keep the reconstructed edit fixed. Use no strength search or prompt rewriting to make the validation outcome look better. Any repair motivated by these cases creates development data and needs a new untouched evaluation set.

## What the original proposal still requires

The original `research_idea.md` describes a larger project than this reporting checkpoint. The following are not completed and must not be implied by our current artifacts:

- A refusal-enhanced or defensive intervention and its safety versus over-refusal trade-off.
- Per-prompt residual or recurrent-state captures, held-out probes and layer-wise representation analysis. Saved means and vectors cannot substitute for those captures.
- Quantized direction re-extraction and cross-precision transfer. Applying the frozen original adapter on a quantized backbone is a different condition.
- The full comparison between quantizing an edited model and extracting and applying a new edit after quantization.
- Multiple precisions or quantizer mechanisms, independent model-family confirmation, or a validated predictive relationship between geometry and behavior.

If the panel expects these original objectives now, we are substantially earlier than a final report. The first checkpoint should be presented as a scoped milestone rather than a completed version of that full program. Existing prior work also prevents treating an observed order difference or cosine change alone as a novel result.

## Mapping to the actual sample

The sample is an 18-page rendered scaffold with many empty chapter pages and unrelated agricultural example material. Its contents page has placeholder numbers extending to page 95; these are not a verified page requirement. It uses an A4-like page size, Times New Roman, centered chapter openings and numbered references. Margins and numbering vary across eighteen sections, so a final report needs deliberate template handling rather than blindly copying its inconsistencies.

| Sample component | Content for this project |
| --- | --- |
| Title and institutional page | Approved project title, actual students and IDs, supervisor, department and submission date |
| Viva approval and acknowledgements | Real institutional wording and genuine acknowledgements; approval/signature fields left for the authorized people |
| Abstract | The sample explicitly asks for 250 to 300 words and five to six keywords; state the actual method, results and limits |
| Contents and lists | Update real chapter/page numbers, tables, figures, algorithms and abbreviations |
| Chapter 1 Introduction | Motivation, problem, scoped questions, objectives and project contribution |
| Chapter 2 Related work | Refusal-direction intervention, hybrid model context, semantic safety evaluation and relevant quantization work |
| Chapter 3 Methodology and implementation | Difference of means, harmless-mean projection, writer sites, fixed adapter, controlled conditions and reproducibility |
| Chapter 4 Results and discussion | Implementation audit, paired semantic results, benign results, marker disagreement, NF4 pilot, failures and limitations |
| Chapter 5 Conclusion and future scope | Answers supported by measured evidence; unfinished enhanced-control, representation and generalization studies |
| References and appendices | Verified relevant bibliography, frozen settings, detailed metrics and output identifiers; private review evidence kept traceable |

Use pipeline and evaluation diagrams, a condition matrix, paired semantic/utility plots, marker-versus-semantic disagreement, and a small set of fully explained cases. Figures must come from saved measurements. Do not add simulated geometry plots or invented benchmark scores to fill pages. Keep operationally harmful details out of the main narrative where a faithful paraphrase and evidence identifier suffice.

## Remaining work blocks

1. **Resolve the model audit.** Record the implementation, metric and semantic verdicts independently. Current owner is `independent_abliteration_audit`.
2. **Complete scoped validation and human review.** Use the audit's prespecified sample for this deadline, review the saved outputs, and document reviewer disagreements and unresolved answers. No extra broad validation batch is planned for tomorrow.
3. **Write the report and assemble evidence.** Draft around the actual result, verify citations, build figures and link every table to saved records. Preserve failures and null findings.
4. **Produce and verify the DOCX and demonstration.** Follow the sample's intended structure, update navigation and references, render and inspect every final page, and provide a reproducible demonstration with recorded versus live outputs clearly identified.

The short-term checkpoint is reached when a panel can inspect the model provenance, rerun the documented procedure, understand exactly what changed behaviorally, and trace the report's conclusions to independent review and saved outputs. A positive outcome must be earned by the evidence. A credible report must explain a negative or mixed outcome just as carefully.

## Sources and evidence

- Retained reference: [`SampleReport.docx`](../SampleReport.docx), SHA-256 `e62bd2c6c049f854c84896c9ff903e4a04d583ca25ec51399f9895474e260f1a`. Read-only XML extraction, metadata and rendered reference pages are retained under `checkpoint_evidence/`; the source is unchanged.
- [`STATE.md`](../STATE.md), [`recovery implementation notes`](../reproduction/qwen35_9b/recovery/IMPLEMENTATION_NOTES.md), and [`feasibility results`](../research/feasibility_v1/RESULTS.md).
- [`original larger proposal`](../research_idea.md) and [`adversarial review`](../research/round2/ADVERSARIAL_TRIAGE.md).
- [Abliterlitics Qwen3.5 9B report](https://abliterlitics.dev/models/qwen3.5-9b/), accessed 10 November 2026. Its 0 of 400 outcome and evaluation settings are external reported claims, not our reproduced findings.
- [Arditi et al](https://arxiv.org/abs/2406.11717) and [StrongREJECT](https://arxiv.org/abs/2402.10260), for established intervention and evaluation context. These sources do not certify our implementation or outcomes.
