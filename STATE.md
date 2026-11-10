# Experiment state

Updated: 2026-11-10 (Asia/Kolkata). Maintain after meaningful changes; keep measurements separate from interpretation.

## Current status

**Implementation, bounded semantic audit and NF4 pilot are complete.** The reconstruction has real but incomplete semantic effects and matches publisher judgment categories on the selected cases; broad efficacy remains unestablished. All GPU jobs are complete. All model execution, dependency installation and full-weight downloads stayed on Kaggle; no local model environment was created. Resume context is in [HANDOFF.md](HANDOFF.md).

**GitHub persistence complete:** private [observer04/refusal-direction-quantization-study](https://github.com/observer04/refusal-direction-quantization-study), branch `main`. The revised report, editable sources, source notebooks and compact research records are tracked. Remote report blob matches the local file. Credentials, model tensors, raw completion queues/logs, output-bearing notebooks and generated render images are excluded; detailed evidence remains separately persisted on private Hugging Face. The historical Heretic release is a pinned upstream submodule. No model execution or dependency installation was needed for this upload.

## Immediate objective

**Revision pending (10 Nov):** read [report/CRITIQUE_FOR_CODEX.md](report/CRITIQUE_FOR_CODEX.md) first. It freezes the research question (does an abliteration edit survive NF4, and is any loss edit-specific?), ranks the report's problems and supplies six data figures in `report/figures_proposed/`. No new GPU work is required.

**Revised college report is complete and checked.** [Final DOCX](report/Refusal_Direction_Ablation_Report.docx), 26 pages, SHA `149f6b2dc43653a5254d1f7b7d3112344a1b68ed252352c7bda5c00b562d24a8`. Title: *Refusal Direction Ablation and Quantization Stability*, with a focused literature review and exploratory Qwen study. The independent triage agent preserved the 28-page v1 in `report/baseline_v1_20261110/`, cited the abstract's literature framing, strengthened comparative synthesis, shortened operational detail and retained all material result limitations. All 26 final pages were inspected; citation, self-containment and template checks pass. Root independently reviewed the revised prose, abstract, synthesis/results pages and final hash. Unknown student, institution, examination and signature fields remain editable placeholders; pagination must be rechecked after filling them. No new inference or uploads occurred.

The report follows `SampleReport.docx` with restrained undergraduate styling and IEEE-style numbered citations. The twelve source identities are frozen in [bibliography plan](report/BIBLIOGRAPHY_PLAN.md), including Kadadekar (2026 preprint) and Joad et al. (EMNLP 2026). Actual tables/figure/explanations are inside the report. Internal evidence routes remain separately in [claim traceability](report/CLAIM_TRACEABILITY.md); [triage record](report/TRIAGE_AND_FIX.md) and `report/writing_qa/QA_NOTES.md` record grounding and verification. Five distinct safe XSTest items were actually used, not a full benchmark. Submission was stated as **11 November 2026**, with no novel-method requirement. No new algorithm, general quantization mechanism, broad capability preservation or conference-ready novelty is claimed. Refusal enhancement, per-prompt geometry, quantized re-extraction and multi-model generalization remain future work.

Independent audit is complete: Kaggle `ommprakash007/qwen35-independent-model-audit-v1` v1 produced **48 new completions**, using **0.87 quota-hours** (24.33 to 23.46h remaining at the post-run check), within its 2h/84-completion cap. Raw outputs are private at `31b8705cb4db4be82c998eecc43ced2a9a930ca5`. Final review commit **`7f12baad08f0f5296b763f2c689be5731e322544`**, path `research/model_audit_v1/review-v1/`, verified all 135 remote artifact hashes and unchanged checkpoint roots. [AUDIT.md](research/model_audit_v1/AUDIT.md) is final; all 72 review records (24 reused plus 48 new), token journals and blank human fields passed QA. No more GPU generation is planned.

Verified semantic counts on eight fresh safety cases: original 1 clear fulfillment / 1 partial / 6 no; publisher and reconstruction each 6 clear / 2 partial / 0 no. On the six prior matched safety cases, publisher and reconstruction both have 2 clear / 3 partial / 1 no under the audit rubric. These are small selected-case, single-agent judgments with imperfect masking; accuracy remains unverified and truncation substantial. They support a real, incomplete behavioral effect and no observed reconstruction deficit on these cases, not general non-refusal or 0/400 reproduction. Earlier feasibility labels remain a separate analysis.

The exact public Abliterlitics v3 heuristic misclassifies some saved redirects as nonrefusal. A CPU witness also demonstrates a closing-thinking-tag preprocessing bug, but unavailable historical raw answers/settings/overrides prevent attributing the external 0/400 result to it. The external report names the same publisher repository. One auxiliary preregistration listing was hashed empty; the unchanged substantive fixtures/source and prelaunch listing derivation are verified, and the packaging exception is being disclosed. No edit optimization or quantizer sweep was run.

| Condition | Marker refusals /100 | Benign first-token KL |
| --- | ---: | ---: |
| Original Qwen3.5-9B | 100 | 0 |
| Failed raw-direction replay | 72 | 0.434047 |
| Actual publisher checkpoint, FP16 | 8 | 0.078430 |
| Corrected fresh-extraction reconstruction | 8 | 0.078022 |
| Publisher's card, reported | 6 | 0.0793 |

## Recovered discrepancy

Our failed replay omitted the upstream harmless-mean orthogonalization step. The corrected recipe forms the difference of means, projects each slot's refusal vector orthogonally to its harmless mean, renormalizes, then interpolates slots 19/20 at 0.49 and applies the published component weights. No optimizer search was needed.

Evidence: sampled publisher weight changes match the expected 19 attention/23 MLP targets. MLP and full-attention delta-direction fingerprints align with projected vectors at cosine >0.9999. A controlled raw/projected comparison changed development marker refusals from 20/32 to 0/32 and benign KL from 0.437735 to 0.065908, with all other settings fixed. A subsequent fresh 400+400 extraction reproduced the pilot's means/directions exactly and recovered the 100-case result above.

Saving and freshly reloading the corrected merged checkpoint gave **100/100 identical responses**. Independent checks verified source/model/dataset pins, prompt order, all 300 original/edited/reloaded marker labels, 100 saved KL terms, vector derivation and all 64 saved adapter B tensors (42 active). The author's exact settings/environment remain unpublished; this is approximate measured reproduction, not byte-identical author weights.

## Evaluation limits

The historical 100 prompts were exposed during recovery. Evaluation uses greedy non-thinking generation, EOS stopping and a 100-token cap; 99/100 corrected responses hit the cap. Benign KL covers the first generated token, not broad capability. The frozen pilot additionally measured 2/64 marker refusals on a deduplicated standard-only subset, not full HarmBench.

On 12 fixed paired 512-token diagnostics, original/corrected marker counts were 11/0; cap counts were 2/11. One-agent qualitative review found semantic withholding in all originals and two corrected answers, with seven corrected answers visibly assisting and three ambiguous. Reviewer labels and raw outputs are retained separately. **No population harmful-compliance rate, 0/400 HarmBench result, broad capability preservation or novel safety-mechanism claim is established.**

## Artifacts and compute

- Start with [Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb](Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb), a self-contained Kaggle T4 ×2 notebook. Detailed evidence: `reproduction/qwen35_9b/recovery/`.
- Corrected full checkpoint: private [Observer04/Qwen3.5-9B-heretic-corrected-20261109](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-corrected-20261109). Remote weight index and all six shard LFS hashes verified in `reproduction/qwen35_9b/recovery/core_persistence_metadata.json`; compact upload proof and final repository revision are in `corrected_persistence.json`.
- Unchanged publisher reference and failed replay remain separately preserved in private `Observer04/Qwen3.5-9B-heretic-validated-20261108` and `Observer04/Qwen3.5-9B-heretic-replay-20261107`. Historical notebooks/results remain intact.
- Recovery GPU jobs: `ommprakash007/qwen35-recovery-projected-pilot` v1 and `ommprakash007/qwen35-recovery-corrected` v1, serial, both COMPLETE. Total measured quota **2.11h** within the 3h envelope; **25.37h remaining**, resets 2026-11-14. No further GPU run is required. A separate optional CPU-only audit was last observed queued, with GPU disabled and a one-hour runtime cap.
- Credentials were excluded from source/output uploads; private runtime input supplies them. No local venvs or model execution.

## Deferred

**Broad quantization studies, repairs and subsequent experimental phases remain deferred.** Following the reviews below, the user authorized only the bounded semantic/quantization feasibility task recorded next. ARA and broad benchmark sweeps are also deferred. The older Daredevil run remains a pilot and does not establish an orthogonalization bug.

## Research question review

Three literature/design reviews are complete. Independent A recommends causal attribution of edit-specific rounding; independent B recommends harmless-mean constraint transfer; adversarial C favors a narrower A as a conditional pilot and B as a diagnostic. Prior work already covers geometry/transfer and practitioner edit-preserving conversion. The proposed increment is causal behavioral attribution plus prediction on independent model/quantizer conditions, not a new projection formula. Precise recommendation and extension requirements: `research/RECOMMENDATION.md`; source reviews: `research/recommendation_{a,b,c}.md`. Conference potential remains unproven. This was research planning only; no new GPU experiments or quantization jobs were launched.

Round 2 is complete: one researcher (`research/round2/RESEARCH.md`, `SEARCH_NOTES.md`), followed by one separate adversarial reviewer (`ADVERSARIAL_TRIAGE.md`). **Neither proposal is selected as a conference-oriented objective.** Prior cache-repair work substantially overlaps control revocation; a practical selective repair beating clean-boundary replay remains unproven. The quantization predictor likewise lacks prospective decision evidence beyond existing reconstruction/floating-edit alternatives. The triage records exact causal controls, implementation checks, stopping criteria and evidence needed to reconsider either branch; another broad literature round is not the next step.

## Bounded feasibility task

Interpretation clarification: feasibility HTML mixes two unedited controls and three edited conditions; it contains 12 prompts repeated across five arms. Historical 8/100 marker refusals do not imply 92/100 substantive harmful completions. The subsequent direct actual-publisher comparison is complete above and found the same selected-case limitations in publisher and reconstruction.

Complete: one execution agent ran the fresh paired baseline and all five conditions on 12 deduplicated prompts (six safety, three sensitive benign, three arithmetic), **60 completions total**. Native NF4 packing, the frozen FP32 adapter on the same packed backbone, all 64 corrected merge hashes and every token journal passed verification. No directions or strengths changed. Native QD refits scales; this is not a fixed-grid rounding experiment.

Single-agent blinded review found two clear visible assistance cases in each corrected arm and none in original FP16/NF4; partial, ambiguous and capped answers prevent a directional interaction conclusion. Label bounds are [−0.500,+0.667], widening to [−0.833,+0.833] with unseen continuations. No observed overrefusal on the three sensitive benign cases; arithmetic was 3 correct per arm except quantized-base-plus-edit (2 correct, 1 capped unknown). These small selected cases establish neither population rates nor broad capability preservation. No clear edit-erasure failure supports repair/predictor development.

All completions, exact prompts, token IDs/settings, raw outputs and cap status are saved, with [paired](research/feasibility_v1/runs/run-v1/output/feasibility_v1/results/HUMAN_REVIEW_ALL.html) and [blinded](research/feasibility_v1/runs/run-v1/output/feasibility_v1/results/HUMAN_REVIEW_BLINDED.html) human review bundles. Human fields are blank; mapping and agent labels are separate. [Results and limitations](research/feasibility_v1/RESULTS.md). Private review-artifact commit `9e4f047bafd49994ee488eec31888d16d43a571f` under `research/feasibility_v1/review-v2/` verified 87 hashes and unchanged checkpoint-root files. GPU job `ommprakash007/qwen35-semantic-nf4-feasibility-v1` v1 is COMPLETE; measured spend **1.04 quota-hours**, **24.33h remaining** (rechecked 10 November; resets 14 November), within the 2h/160-completion ceiling. No inference rerun. Independent human review of saved outputs is the next useful step; repairs, broader sweeps, conference claims and cache revocation remain deferred.
