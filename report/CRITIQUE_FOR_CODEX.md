# Critique of the final report draft and the frozen research question

Reviewed 10 November 2026 against `report/Refusal_Direction_Ablation_Report.docx` (26 pages, SHA `149f6b2d…`), using its extracted text `report/writing_qa/report_text.txt`, plus `STATE.md`, `HANDOFF.md`, `research_idea.md`, `research/RECOMMENDATION.md` and `research/round2/ADVERSARIAL_TRIAGE.md`. Submission is **11 November 2026**. Nothing below needs new GPU work. Every number in the proposed figures is copied from verified tables already in the report or in `research/model_audit_v1/`.

**Verdict.** The evidence is good and the honesty is good. The report still reads badly because it has **no spine**: the title promises a quantization-stability study, but the body is three loosely joined mini-studies (a reproduction, a metric audit and a 6-prompt NF4 pilot), followed by "what a future study would need". More than half the paragraphs end in a disclaimer, so a reader finishes without knowing what was found. The fix is mostly framing and editing, not new content. Freeze one question (§1). Recast the reproduction and the metric audit as the two preconditions for answering it. Lead each section with its finding. Gather the caveats into one place. Add figures.

---

## 1. The frozen research question

### The question

> **When a refusal-ablation ("abliteration") edit is baked into a model's weights at full precision and the model is then quantized to 4 bits for deployment, does the edit's behavioural effect survive, judged by what the model actually says rather than by refusal phrases? And can loss of the edit be told apart from ordinary quantization damage?**

Suggested title: **Does Abliteration Survive 4-bit Quantization? A Controlled Reproduction and Semantic Evaluation on Qwen3.5-9B**. A title phrased as a question tells the reader what the report is about before they read anything else.

### Why this question and not another

1. **It is the team's original idea, narrowed.** `research_idea.md` was "baseline → abliterate → quantize → compare", then a commutativity question, `Q(A(M))` vs `A(Q(M))`. The frozen question is the half of that idea the project actually built: edit first, then quantize. It is also the half that matches how people really deploy these models (next point). `A(Q(M))`, re-extracting the direction from a quantized model, becomes clearly scoped future work instead of a silent omission.
2. **It matches how these models are deployed.** Abliterated checkpoints are made in BF16/FP16 and then mostly downloaded as 4-bit GGUF/NF4/AWQ quants. Whether the edit survived the conversion is rarely checked. The practical question is "is the 4-bit file I downloaded still abliterated?"
3. **It fills a gap in the literature.** Quantization-safety work (Q-resafe [4], Kadadekar [12], Kumar et al.) studies what quantization does to the *original* alignment of a model. Egashira et al. (NeurIPS 2024) show quantization can *create* behaviour that is absent at full precision. Arditi [1], Heretic [5] and grimjim's projected abliteration study the edit at full precision only. A post-hoc, low-rank weight edit passing through rounding sits between these lines of work. The only direct evidence so far is practitioner model cards reporting edit loss under FP4 conversion (the lflare NVFP4 card already noted in `research/RECOMMENDATION.md`).
4. **The existing evidence can answer it at pilot scale.** The five-arm NF4 design (`W`, `W+Δ`, `Q(W)`, `Q(W+Δ)`, `Q(W)+Δ`) is exactly the controlled design this question needs. It is already run, verified and reported (Table 4.3).
5. **It unifies the three current RQs.** You cannot ask whether an edit survives quantization until you have (a) a *faithful* edit and (b) a *valid* measure of its effect. The reproduction chapter and the metric-disagreement chapter therefore stop being side quests: they are the two preconditions. This is the single biggest improvement available.

### Sub-questions (replace RQ1–RQ3 in §1.3)

| Sub-question | Role | Answer the existing evidence supports |
| --- | --- | --- |
| **SQ1. Can a published abliteration be reproduced faithfully enough to study it?** | Precondition: a known Δ | Yes, approximately. The first replay failed (72/100 marker refusals, KL 0.434). Restoring the harmless-mean projection, which is grimjim's "projected abliteration" adopted by Heretic, gives 8/100 and KL 0.078, matching the publisher's checkpoint (8/100, 0.078). The edit was verified matrix by matrix, and reloading gave 100/100 identical outputs. |
| **SQ2. Does "no refusal phrase" measure the edit's effect?** | Precondition: a valid outcome H | No, not on its own. Marker matching, the public v3 heuristic and semantic judgment disagree on the same outputs: a redirect counts as "non-refusal", and refusal wording co-occurs with fulfilment. Semantic fulfilment is the outcome used for SQ3. |
| **SQ3. Does the edit's effect survive NF4, and is any loss edit-specific?** | The question | On this model and NF4 configuration, no erasure was observed. Every clear fulfilment seen in FP16 (2/6) is still present after quantizing the edited model, and after keeping the edit in floating point on the quantized base. The unedited quantized model stays at 0/6, so quantization alone did not produce the effect. The pilot is too small, and has too many partial/capped answers, to rule out *partial* weakening. |

**One-sentence answer the abstract and conclusion should state plainly:** *On Qwen3.5-9B with native NF4, a faithfully reproduced abliteration edit kept its observed behavioural effect after quantization, but measuring that required fixing the reproduction first and replacing refusal-phrase counting with semantic judgment; whether the effect is partly weakened remains open at this sample size.*

That sentence is honest, it is a result, and it gives the reader the "aha": the hard part of the question turned out to be measurement, not compute.

### A mechanistic hypothesis to add to the discussion (labelled as untested)

Give the reader a reason to *expect* survival or loss. One paragraph, no new data. Each edited matrix gets a rank-1 update `ΔM = −λ r̂ (r̂ᵀ M)`. Its energy lies entirely along one output direction r̂. Round-to-nearest NF4 error is roughly unstructured across a 4096-dimensional output space, so only about 1/d of its energy falls along r̂. The edit is therefore coherent signal and quantization error is diffuse noise. This predicts that blockwise round-to-nearest quantization should *preserve* a rank-1 abliteration edit. Loss is more plausible for coarser or microscaled formats (FP4 with shared scales, very low bit-widths) or for edits with small λ. That fits the pilot, and it explains why the practitioner FP4 report might differ. **Present it as a hypothesis, not a finding.** The cheap test is weight-space only and runs on CPU on Kaggle: compute the retained-edit fraction `ρ = ⟨Q(W+Δ) − Q(W), Δ⟩ / ‖Δ‖²` per matrix. Make that the first item of future work (§5.2). It turns "future work" from a list into a concrete next experiment.

### Out of scope (one short list, stated once)

Re-extraction after quantization (`A(Q(W))`), other quantizers and bit-widths, a second model family, refusal-*enhancing* edits, activation-level geometry, the harmful accuracy of outputs, population rates. Retire the KV-cache revocation idea from round 2 completely. It is a different research programme and should not appear in this report.

---

## 2. Why the direction felt fuzzy

The project changed its framing four times without retiring the previous one, and the report carries traces of all four:

1. `research_idea.md`: "does quantization rotate the refusal direction?" (geometry, cosines, probes). None of this was measured.
2. Commutativity, `Q(A(M))` vs `A(Q(M))`. Only the `Q(A(M))` side was built.
3. `RECOMMENDATION.md`: "edit-specific quantization error *predicts* behavioural loss, plus a targeted repair". There is no predictor and no repair.
4. Round 2: KV-cache "revocation". Rejected.

Meanwhile, the work actually done was reproduction plus semantic audit plus a five-arm pilot. The report's Chapter 2 §2.6 and §4.6 still pitch framing 3 (prediction and diagnostics) as "the question pursued beyond this baseline". That is why the report reads as a preamble to a study that does not exist. **Fix: write the report about the question in §1, which the evidence answers, and mention framings 1–3 only as future work.**

---

## 3. Ranked critique with concrete fixes

### C1. No spine (critical)
- §1.2 ends with "The report asks what can be established now and what a stronger deployment study would need to test." That is not a research question.
- RQ1–RQ3 (§1.3) are parallel and unconnected. RQ3, the one the title is about, gets the smallest section (§4.4) and is "answered only at the feasibility level".
- **Fix:** adopt §1 above. Rewrite §1.2–1.3 around the frozen question and SQ1–SQ3. Rename chapters/sections so each one answers one SQ.

### C2. Hedge density buries the findings (critical)
- Measured: **35 of 62** substantial paragraphs in Chapters 1–5 contain a disclaimer or negation, about 68 negation tokens in about 4,000 words of body text. Examples: abstract ¶2 ends "These are phrase-marker and persistence results, not verified harmful-compliance rates."; §4.3 ends three consecutive paragraphs with a "does not establish" clause; §4.6 is nearly all disclaimer.
- **Fix:** each results paragraph states the finding in its first sentence and allows at most one limitation sentence. Move every other caveat into §4.7 "Threats to validity". Add a small two-column box/table at the end of Chapter 4: "What this study shows / What it does not show". Do not delete any material limitation. Consolidate them. Readers trust a report more when its caveats sit in one clear place.

### C3. The abstract leads with a number the report then disowns (high)
- Abstract ¶2's headline is "72 → 8 refusal-marker matches", followed by an immediate statement that it is not a harmful-compliance rate. The reader learns the wrong number first.
- **Fix:** rewrite the abstract as five moves: problem, question, method, findings, implication. Suggested draft (about 230 words; adjust to taste):

> Abliteration removes a model's refusal behaviour by projecting a "refusal direction" out of its weights, and abliterated models are commonly distributed in 4-bit quantized form. Whether the edit survives that conversion is rarely checked. We ask: when a refusal-ablation edit is applied at full precision and the model is then quantized to NF4, does its behavioural effect survive, and can edit loss be separated from ordinary quantization damage? Answering this required two preconditions. First, we reproduced a published Heretic edit of the hybrid Qwen3.5-9B model. An initial replay failed until we restored a harmless-mean projection step ("projected abliteration"), after which our reconstruction matched the publisher's checkpoint in refusal-marker rate (8/100) and benign KL divergence (0.078) and was verified matrix by matrix. Second, we show that refusal-phrase counts are not a valid outcome: two automated refusal detectors and a semantic fulfilment review disagree on the same outputs. Using semantic judgments, both edited models clearly fulfilled 6 of 8 fresh safety requests, compared with 1 of 8 for the original. Finally, a controlled five-arm NF4 comparison found no case in which quantization erased an observed edit effect, while the unedited quantized model remained unchanged. The samples are small, judgments come from one LLM-based reviewer, and many answers are truncated, so partial attenuation cannot be ruled out. We outline a weight-space test of why rank-1 edits should survive round-to-nearest quantization.

(Keywords must include **abliteration**.)

### C4. The most interesting result is under-told and uncited (high)
- The word **"abliteration" appears 0 times** in the report, although it is the name of the technique and the term every reader will search for. Add it to the title or subtitle, the abstract, the keywords and Chapter 2.
- The "omitted upstream harmless-mean projection" (§3.2, §4.1) is a known, published technique: grimjim's **Projected Abliteration** (Hugging Face blog, 25 Oct 2025), later extended to norm-preserving biprojected abliteration. The report presents it as an anonymous Heretic implementation detail. Recast the finding as a **controlled ablation of a known design choice on a hybrid architecture**: with strengths and layers fixed, projection alone moved markers from 72 to 8 per 100 and KL from 0.434 to 0.078 (dev check 20/32 → 0/32, KL 0.438 → 0.066). That counts as a contribution, and it is currently hidden.
- Explain *why* projection helps (one paragraph plus Fig. 2). The raw difference of means contains a component along the general "any prompt" mean direction. Ablating that component perturbs every input, which raises KL. It also spends part of the unit-norm direction on something other than refusal, which weakens refusal removal at fixed λ. Label this as the standard explanation, cite grimjim, and do not claim it was measured mechanistically.

### C5. Agent and process jargon leaks into the prose (high)
These phrases mean something to the agents that built the project and nothing to an examiner. Rewrite or define each one at first use:

| Current phrase | Problem | Replace with |
| --- | --- | --- |
| "exposed historical 100-prompt test" | "exposed" is internal | "the 100-prompt test set, which we also used while debugging (so it is not a clean held-out set)" |
| "fresh" / "matched" cohorts | used before definition | define once: "matched = 6 prompts reused from the NF4 pilot; fresh = 8 new prompts chosen before any outputs were read" |
| "24 gated reuses" | process detail | "24 earlier responses reused after confirming identical settings" (or move to appendix) |
| "A timing rule selected complete prompt families" | opaque | "Prompts were fixed before any new outputs were inspected" |
| "freezes each study's labels before opening its condition mapping" | opaque | "labelled anonymised, shuffled outputs before learning which model produced each" |
| "harness" | jargon | "our evaluation setup" |
| "slots 19 and 20", "source coordinates rather than ranked tutorial candidates" | opaque, and "tutorial" is out of context | "the direction is taken from the residual stream after layers 18–19 (interpolated 49%)"; delete the tutorial clause |
| "One automated reviewer" | **undisclosed AI use** | "a single LLM-based reviewer (AI-assisted annotation; no human labels)". See C11. |

### C6. Related work is thin, padded in one place and missing core papers (medium-high)
- §2.1 spends its space and Eq. 2.1 on scaled dot-product attention, which no later argument uses. Replace it with the residual stream as a sum of block writes, the reason a writer-matrix projection removes a direction, and the hybrid Gated DeltaNet / full-attention layout (Fig. 3).
- There are 12 references, which is light for a report billed as a "literature review". Add these (✓ = verified this session; ◻ = verify the bibliographic details before citing):
  - ✓ Egashira, Vero, Staab, He, Vechev, "Exploiting LLM Quantization", NeurIPS 2024 (arXiv 2405.18137). Quantization can *activate* behaviour that is absent at full precision. This is the closest conceptual neighbour to "does an edit survive quantization".
  - ✓ grimjim (J. Lai), "Projected Abliteration", Hugging Face blog, 25 Oct 2025, and "Norm-Preserving Biprojected Abliteration", 6 Nov 2025. Both are directly relevant to C4.
  - ✓ Wollschläger et al., "The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence", arXiv 2502.17420 (2025). Refusal is multi-dimensional; check the venue before writing "ICML".
  - ◻ M. Labonne, "Uncensor any LLM with abliteration", Hugging Face blog, 2024. This is the origin of the term and of the `mlabonne/harmful_behaviors` dataset already used.
  - ◻ Zou et al., "Representation Engineering", arXiv 2310.01405 (2023); Turner et al., activation addition (2023). These give the steering lineage.
  - ◻ Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep", ICLR 2025. Relevant to why refusal edits act early in generation.
  - ◻ Kumar et al., "Fine-tuning, Quantization, and LLMs: Navigating Unintended Outcomes", arXiv 2404.04392.
  - ◻ Xie et al., SORRY-Bench (2024), as an alternative refusal benchmark, if space allows.
- The literature table (Table 2.1) is fine. Add a final row for "this report", stating the gap it fills.

### C7. Almost no figures, and no data plots at all (high)
- The only figure (3.1) is a box diagram that restates the text. `HANDOFF.md` explicitly asked for measured plots, but none exist. Six figures are ready in `report/figures_proposed/` (see §5). They are generated by `make_figures.py` from verified numbers, in a serif font at 300 dpi.

### C8. Notation and table consistency (medium)
- `D` and `W + Δ` both name the edited model. `W` is both the model and a weight matrix (Eq. 3.2 uses `M` for the matrix, which helps). Define once, in a notation table in §3.1: `W` = original model, `Δ` = edit, `D ≡ W+Δ`, `P` = publisher, `Q(·)` = NF4 conversion, `H(·)` = fraction of clear fulfilments. Then use those symbols everywhere.
- Table 4.3 orders its columns Clear / No / Partial / Unknown, while Table 4.2 uses Clear / Partial / No. Use the same order (Clear / Partial / Unknown / No) in both.
- Table 4.1: the original model's "Capped 59/100" is never discussed. Either explain it in one clause or drop the column and say "99/100 edited answers hit the 100-token cap" in the text.
- Table 3.2 row "100 / 100 benign KL prompts / 0" is hard to parse. Split the KL prompts into their own row.
- Table A.1: in the extracted text, the revision hash runs straight into the role ("…337b9aOriginal model"). Check that the cell has a line break or a separate column in the DOCX.

### C9. The interaction-bound paragraph is impenetrable (medium)
- §4.4 ¶3 (the `[H(Q(W+Δ)) − H(Q(W))] − [H(D) − H(W)]` bounds from −0.500 to +0.667, widening to ±0.833) will lose every examiner. **Fix:** in the main text, show Fig. 1 and Fig. 5 and say: "The edit raises clear fulfilments from 0 to 2 of 6 in FP16 and from 0 to 2 of 6 after NF4, so the observed edit effect is the same size. Because several answers are partial or truncated, the true difference could lie anywhere from −0.5 to +0.67." Move the formal definition and the censoring bounds to an appendix.

### C10. Missing ethics and dual-use statement (medium-high for a college report)
- A report on removing safety refusals needs a short explicit section, for example §1.5 or §3.6, "Ethical considerations". It should cover the purpose (understanding the robustness of safety edits under deployment transformations), state that no harmful output text is reproduced, that raw outputs and the reconstructed checkpoint are kept private and access-controlled, that harmful accuracy was deliberately not tested, and that only publicly released edits were reproduced. At the moment this appears only as one sentence in Appendix A.

### C11. AI-assistance disclosure is missing (high: academic integrity)
- `HANDOFF.md` requires clear disclosure of AI-assisted implementation and annotation. The report only says "one automated reviewer". Add a short statement (methodology §3.5 or the acknowledgements) that the implementation, orchestration and semantic labelling were AI-assisted, that the labels come from an LLM reviewer, and that no independent human labels exist yet. The student should adjust the wording to the institution's policy.

### C12. Conclusion and future work have no priority order (medium)
- §5.2 lists everything a future study should do. **Fix:** (1) answer the frozen question in one paragraph using the one-sentence answer from §1; (2) give three prioritised next steps: (a) the CPU-only weight-space retained-edit fraction ρ across bit-widths, which tests the hypothesis in §1; (b) human review of the 132 saved outputs (72 + 60); (c) a larger semantic evaluation with a calibrated judge (StrongREJECT-style) and a second quantizer, then `A(Q(W))` re-extraction.

### C13. Smaller items
- The abstract is 269 words, which is within the 250–300 template range. Keep that range after the rewrite.
- The heading "2.2 From linear control to a richer representation" is vague. Use "2.2 Refusal directions: single direction or many?"
- The external 0/400 Abliterlitics claim plus the thinking-tag parser witness is a good, self-contained story about metric validity. Keep it in §4.5, but end it on the finding ("leaderboard refusal rates for abliterated models depend on the detector") rather than on "no historical cause is established".
- The Figure 3.1 caption says arrows "express study logic, not execution dates". Delete that. Replace Fig 3.1 with the new Fig. 1.

---

## 4. Proposed structure (same length, about 26 pages)

1. **Introduction**: motivation (edits are made in FP16 and deployed in 4-bit); the frozen question; SQ1–SQ3 as "two preconditions plus the question"; ethics statement; contributions as three bullets.
2. **Background and related work**: residual stream and abliteration (Arditi, Labonne, Heretic, grimjim projection); refusal geometry (Joad, Wollschläger); evaluating refusal (HarmBench, StrongREJECT, XSTest); quantization and safety (QLoRA/NF4, Q-resafe, Egashira, Kadadekar, Kumar); the gap, with Table 2.1 plus a "this report" row.
3. **Method**: model and edit map (Fig. 3); projection (Fig. 2, Eq. 3.1–3.2); the five-arm design (Fig. 1, Table 3.1, notation table); outcome measures (Fig. 6 claim ladder); samples and generation settings (Table 3.2); verification; AI-assistance statement.
4. **Results**: §4.1 SQ1, reproduction (Table 4.1, Fig. 2 right panel); §4.2 SQ2, metrics disagree (Fig. 4, Table 4.4, the 0/400 and parser story); §4.3 SQ3, survival under NF4 (Fig. 5, Table 4.3, plain-language effect statement); §4.4 why the edit might survive (the hypothesis, clearly labelled); §4.5 threats to validity, with every caveat in one place plus a "shows / does not show" box.
5. **Conclusion**: the one-sentence answer, then three prioritised next steps.
- Appendix A: revisions and recipe (as now), the interaction bounds, the fixture list.

---

## 5. Figures, ready to insert (`report/figures_proposed/`)

Regenerate with `python3 report/figures_proposed/make_figures.py`. All data is copied from verified sources and named in the script docstring. Each figure needs a caption that states the takeaway, not only the content.

| File | Place | Suggested caption |
| --- | --- | --- |
| `fig1_question_design.png` | §1.3 or §3.1, replacing Fig 3.1 | **The research question as a controlled comparison.** The edit's effect is measured twice, at full precision (top) and after NF4 (bottom), and the two are compared. The unedited NF4 model separates generic quantization damage from loss of the edit. Counts are clear fulfilments in the 6-prompt pilot. |
| `fig2_projection_geometry.png` | §3.2 / §4.1 | **Projecting out the harmless mean is what made the reproduction work.** Left: schematic. Right: with strengths and layers fixed, projection alone cut refusal-marker matches from 72 to 8 per 100 and benign KL from 0.434 to 0.078. |
| `fig3_edit_map.png` | §2.1 / §3.2 | **Where the edit lands in Qwen3.5-9B.** Top: the 3:1 linear-attention / full-attention layer pattern. Bottom: per-layer ablation strength λ for attention output projections (19 layers) and MLP down projections (23 layers), computed from the published Heretic parameters. λ > 1 over-ablates (reverses the component). |
| `fig4_semantic_vs_markers.png` | §4.2 | **Three measures, same outputs.** Semantic judgments (bars) compared with two automated refusal detectors (right columns). Zero detected refusals coexist with partial fulfilment, and refusal wording coexists with fulfilment. |
| `fig5_nf4_pilot.png` | §4.3 | **The edit's observed effect survives NF4.** Both quantized-edit arms keep the two clear fulfilments seen in FP16, and the unedited NF4 model matches the original. Hatched bars are unresolved answers, and most edited answers hit the token cap. |
| `fig6_claim_ladder.png` | §3.5 or §2.3 | **Four claims that are often conflated.** Each rung needs stronger evidence. This report establishes rungs 1–2, gives selected-case evidence for rung 3, and deliberately leaves rung 4 untested. |

Before inserting, Codex should check that the figures render at text width without labels shrinking below about 8 pt, and update the List of Figures.

---

## 6. Guardrails: what must not change

- Do not add any new number. Every figure and the abstract draft reuse already-verified values. Do not upgrade "selected cases" to rates, and do not drop the truncation, single-reviewer or small-sample caveats. Consolidate them instead (C2).
- No new GPU runs. The weight-space ρ test is future work unless the user explicitly authorises a Kaggle CPU job before submission.
- Keep the student, supervisor and signature placeholders. Do not invent acknowledgements or approvals.
- Preserve the current DOCX as `report/baseline_v2_20261110/` before editing, following the existing v1 baseline pattern.
