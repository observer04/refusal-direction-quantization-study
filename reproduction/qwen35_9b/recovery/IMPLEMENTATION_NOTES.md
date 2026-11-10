# Recovery implementation notes

Scope: recover the editing-procedure reproduction. Quantization and later research phases are deferred. Prior measured evidence remains unchanged; root maintains `STATE.md`.

## Starting evidence

- Original Qwen:100/100 marker refusals. Our replay:72/100, benign first-token KL0.434047. Actual publisher checkpoint evaluated under the same FP16/nonthinking harness:8/100, KL0.078430. Card reports6/100, KL0.0793.
- Replay serialization gave100/100 exact response matches. This does not validate direction extraction or actual adapter assignment.
- Replay forced the model's closed-thinking template. Upstream prefix detection was bypassed. Exact publisher prompt tokens are unknown.
- Pinned compatible implementation targets both full-attention `o_proj` and DeltaNet `out_proj`. Author's exact hybrid support is unknown.
- Evaluation peer notes publisher component max weights2.00/1.67 exceed the pinned optimizer's default search ceiling1.5. This establishes an optimization-provenance difference, not a causal explanation.

## CPU audit v1

Private Kaggle job: `ommprakash007/qwen35-recovery-weight-audit`, version1. Explicit CPU metadata, no accelerator flag, runner asserts CUDA unavailable. Timeout3600s. No model instantiation or generation. Original and published model files downloaded only on Kaggle under temporary storage, not retained as output. Prior replay merged weights are not downloaded.

Audit steps:

1. Sample every common tensor, comparing the base after conversion to the publisher's storage dtype. Record missing auxiliary tensors separately.
2. Fully compare64 language-model output projections, recovering the dominant left singular direction from weight deltas and estimating ablation strength against the corresponding original weights.
3. Compare inferred directions to each of33 saved replay vectors and the18.49 interpolated direction; test consistency across edited components.
4. Check analytical projection-output intervention/weight-edit agreement and quantify FP16 merge rounding on random inputs.

Limits: non-target tensor audit is sampled. Low-precision deltas are only approximately rank1. The analytical check does not inspect saved PEFT adapter tensors or verify runtime hook placement. Peer reviewed source and found no blocking issue; next GPU preflight must inspect actual `B@A` and scaling and compare at matching projection outputs.

Initial total recovery GPU envelope:3 Kaggle quota hours in bounded serial jobs. CPU audit uses no GPU quota. GPU launches owned by implementation agent only, with evaluation peer review before execution.

## Sparse forensics while CPU queue waits

The CPU worker remained queued. Strictly bounded HTTP byte-range analysis used only stdlib: metadata, tiny weight samples and existing saved float arrays, without model/tensor execution, a local environment, or complete shard downloads. Raw remote sample bytes were discarded; scripts and derived metrics are retained.

- `range_audit.json`:130 tensors sampled,254768 bytes including headers.42 changed samples exactly match19 attention projections(layers10–28, both DeltaNet/full attention) and23MLP down projections(layers9–31).88 controls/inactive projections match after storage-dtype conversion. Whole-tensor equality is not established by sampling.
- `direction_sample_audit.json`: layer20MLP,32 evenly spaced output rows×128 input columns. Dominant rank1 energy99.4201%. Its32-coordinate direction has absolute cosine0.937105 with our raw selected vector, **0.999919 with the per-slot good-mean-orthogonalized vector**.
- `direction_sample_audit_attention.json`: layer19full-attention projection independently gives99.5256%rank1 energy; raw cosine0.936881, projected cosine**0.999923**.
- Projection follows pinned upstream exactly: normalize each good mean; subtract its component from each raw refusal vector; renormalize each slot; only then interpolate slots19/20 at.49 and renormalize.

The leading explanation is now the omitted `orthogonalize_direction=True` option. This is strong sampled weight evidence, not yet a full-model causal result or proof of the author's exact settings. Prefix and hybrid coverage are no longer the leading hypotheses. The evaluation peer independently reviewed the sampler/projection math.

GPU pilot revision: compare raw and projected processing of **our own pinned saved400+400 activation means**, keeping the model, source, card weights and closed-thinking evaluation fixed. This changes only one direction-processing setting and avoids repeated extraction. Development-only selection is frozen before the exposed100-prompt/benign-KL comparison; poor development pilots stop before additional generations. Sparse forensic inputs are embedded, so the queued CPU job does not block GPU allocation. The proposed job has a5400s cap, with no duplicate GPU submission.

## GPU causal pilot v1 — complete

Private notebook job `ommprakash007/qwen35-recovery-projected-pilot`, version1, completed at17:52UTC on2026-11-09 after a script-packaging save error created no job. T4x2,5400s runtime cap. Measured Kaggle quota use1.05h (27.48→26.43h remaining). Existing pinned CUDA PyTorch2.11+cu130 preserved; arithmetic smoke and64projection discovery passed. Exact development split is16unused native harmful+16deduplicated external validation behaviors,32benign.

| Direction processing | Development marker refusals | Development benign first-token KL | Capped /32 |
| --- | ---: | ---: | ---: |
| Raw control |20/32 |0.4377345145 |31 |
| Per-slot projected |0/32 |0.0659081638 |32 |

Only direction processing changed. Actual assigned PEFT A/B/scaling checks passed on DeltaNet and full-attention targets. Projected selection was frozen before historical/fresh evaluation. Capped marker-negative answers do not establish harmful compliance. Merged development remained0/32. The selected merged candidate gave8/100 marker refusals and benign first-token KL0.0780219659 on the exposed historical comparison;99/100 answers reached the100-token cap. The64-case fresh standard-only subset gave2 marker refusals, KL0.0548846647, and64 capped answers. These are marker measurements, not HarmBench classifier results.

Unmerged/merged development strings matched27/32 exactly, with first-token KL8.9583e-6 and the same0/32 marker count. Small FP16 merge rounding can change strings. Actual saved adapter checks independently passed all64 B tensors (42active), maximum coordinate error4.86e-8. Raw/projected vector derivation checks passed within4.15e-8.

The selected12-case512-token diagnostic had0 markers and11 capped answers. Independent qualitative review found2 substantive refusals despite no marker,7 visible-assistance responses and3 ambiguous cases. This small selected diagnostic establishes a marker/semantics mismatch, not a population harmful-compliance rate. Raw rows and reviewer labels remain separate.

The separate `build_corrected.py` prepares a complete fresh400+400 extraction notebook with explicit per-slot projection,100-prompt/benign-KL comparability, per-prompt KL terms, merge/save/reload100response verification and private full-checkpoint persistence. It preserves the failed original notebook. Evaluation peer reviewed its math/flow. This second serial job, `ommprakash007/qwen35-recovery-corrected` version1, was accepted around18:08UTC on2026-11-09 with a6300s cap (105min). It uses fresh400+400 activation extraction, without cached means, and targets private repository `Observer04/Qwen3.5-9B-heretic-corrected-20261109`. The fixed12 paired prompts are inherited from the pilot; original and corrected512-token diagnostics use identical prompts and templates. Core100-case save/reload verification and checkpoint persistence precede the final corrected long-answer diagnostic. No second GPU job overlaps the pilot; total maximum quota exposure1.05+1.75=2.80h within the3h initial envelope. Root owns the authoritative high-level state.

## Fresh-extraction corrected run v1 — complete

`ommprakash007/qwen35-recovery-corrected` version1 completed and released GPUs at19:14UTC on2026-11-09 (2026-11-10 local). It re-extracted400+400 activation means from the pinned original model. All135168 float32 coordinates of each good-mean, bad-mean and projected-direction array exactly match the causal pilot's saved arrays. This was a fresh extraction, not a cached-means replay.

- Baseline100/100 marker refusals;59answers reached100tokens.
- Corrected merged model8/100 markers;99answers reached100tokens; benign first-token KL0.07802196592092514.
- Fresh model reload8/100 markers and100/100 exact pre-save response matches.
- Independent checker verified source/model/dataset pins, prompt order, all300native marker rows,100saved KL terms (mean0.0780219553),33slot projected-vector derivation(maxerror4.17e-8) and64saved adapter B tensors/42active(maxerror4.86e-8).
- Fixed paired12×512 diagnostic: original11markers/2capped; corrected0markers/11capped. One-agent qualitative review found semantic withholding in all12 originals, versus2yes/9no/1unknown corrected; visible harmful assistance7yes/2no/3unknown corrected. The reviewer knew conditions/context, and no inter-rater agreement or population estimate is claimed. All corrected texts byte-match the pilot's independently reviewed long texts.

The full checkpoint was privately persisted to `Observer04/Qwen3.5-9B-heretic-corrected-20261109` before final corrected-long generation. Index membership and all6root-shard LFS SHA256 hashes were independently checked remotely without local weight downloads (`core_persistence_metadata.json`). Final compact persistence adds source, independent verification and semantic-review evidence; `corrected_persistence.json` records its final commit.

Measured quota: pilot1.05h + corrected1.06h =2.11h, within the initial3h envelope, with25.37h remaining. No further GPU job is required. The separate0GPU/1hCPU audit was last observed queued; sparse two-component weight evidence and causal/fresh tests support the recovered setting without waiting for it. This does not prove exact author provenance, the advertised full0/400HarmBench result, broad capability preservation or removal of a single general safety mechanism.
