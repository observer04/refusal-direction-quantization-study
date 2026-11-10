"""Panel-facing report content with a frozen twelve-source bibliography.

The report distinguishes literature findings, measured observations and future work.
Author-side evidence locations are recorded separately in CLAIM_TRACEABILITY.md.
"""
TITLE = "Refusal Direction Ablation\nand Quantization Stability"
SUBTITLE = "Literature Review and an Exploratory Qwen Model Study"

def p(text): return {"type":"p", "text":text}
def h(text): return {"type":"h", "text":text}
def table(caption, headers, rows, widths): return {"type":"table","caption":caption,"headers":headers,"rows":rows,"widths":widths}
def equation(text, number): return {"type":"equation","text":text,"number":number}
def steps(caption, rows): return {"type":"steps","caption":caption,"rows":rows}
def page(key, title, blocks, chapter=None, section=None):
    return {"key":key,"title":title,"blocks":blocks,"chapter":chapter,"section":section}

ABSTRACT = """Linear interventions can strongly alter refusal in language models {arditi}, but refusal wording alone does not establish useful fulfillment of a request {strongreject}. This report combines a focused literature review with a Qwen3.5-9B case study to examine the foundations for evaluating behavioral edits after quantization. The review connects refusal representations, semantic evaluation and quantization safety, and motivates a question about whether a fixed edit retains its intended effect after compression.

We reconstruct a published Heretic edit and compare it with the original model and actual publisher checkpoint under matched settings. Correcting an omitted upstream harmless-mean projection reduces refusal-marker matches from 72 to 8 on the exposed historical 100-prompt test. The publisher also records 8 in this harness, and all 100 reconstructed responses remain identical after saving and reloading. These are phrase-marker and persistence results, not verified harmful-compliance rates.

A separate semantic review compares fourteen selected safety cases. On eight fresh cases, the original receives one clear requested-content fulfillment judgment, while publisher and reconstruction each receive six. Both edited models receive the same fulfillment categories across the fourteen cases. One automated reviewer supplies the judgments with imperfect condition masking; technical correctness is unverified and many outputs are truncated.

A five-condition native NF4 pilot compares unedited and edited floating-point models with quantized and floating-edit alternatives. Two clear edited assistance cases remain visible after quantization, but partial answers and truncation leave its interaction with the edit unresolved. The contribution is a grounded review and reproducible exploratory study. The proposed next direction is prospective testing of edit stability using semantic outcomes and matched controls; a predictive mechanism and general capability preservation remain unestablished."""
KEYWORDS = "large language models; refusal direction; quantization; reproducibility; semantic evaluation; NF4"

PAGES = [
page("intro1", "INTRODUCTION", [
h("1.1 Motivation and research setting"),
p("Refusal direction ablation offers a concrete way to study how a language model's internal representations influence its responses. Arditi et al. show that removing or adding a direction can alter refusal in the chat models they investigate {arditi}. The relevant outcome is behavioral: a useful intervention must be assessed through what the model actually answers, not only through a geometric change to its weights."),
p("Compression introduces a second question. QLoRA makes four-bit NF4 weights practical for memory-efficient adaptation {qlora}, while Q-resafe studies safety changes and patching after quantization {qresafe}. These works motivate examining behavioral preservation separately from ordinary task quality. An edited checkpoint is a useful test case because the intended change can be defined relative to its unedited control."),
p("This report studies a Heretic directional edit {heretic} for Qwen3.5-9B {qwen}. The work combines a focused review with an exploratory reproduction and native NF4 comparison. It treats the reconstruction as the baseline needed for a later investigation of edit stability. The experiment does not introduce a new editing algorithm."),
h("1.2 Problem statement"),
p("Three different claims can be confused: that an edit is numerically implemented, that refusal language decreases, and that the requested content is substantively supplied. StrongREJECT shows why useful forbidden information must be evaluated rather than inferred from non-refusal {strongreject}. A quantization study adds another distinction: a changed answer may reflect general model degradation rather than loss of the particular edit."),
p("Abliterlitics reports zero refusals in 400 cases for the selected publisher variant {abliterlitics}. That external claim motivated our comparison, but its historical settings and raw outputs are insufficiently specified for an exact replay. We therefore use matched controls and report our own measured denominators. The report asks what can be established now and what a stronger deployment study would need to test.")
], chapter=1, section=9),
page("intro2", "", [
h("1.3 Research questions and objectives"),
p("The immediate research questions are RQ1: does the reconstructed edit implement the recovered recipe and resemble the publisher under the tested settings? RQ2: how do phrase-based refusal counts differ from semantic judgments of the answers? RQ3: does a bounded native NF4 comparison reveal a change in the effect of the fixed edit? These questions separate a reproducible baseline from the longer-term goal of predicting edit stability."),
p("The objectives are to synthesize the relevant literature, reconstruct and verify the edit, compare original and edited models using the same inputs, and establish an interpretable quantization comparison. The broader question is whether an edit-specific numerical diagnostic can predict semantic failure after conversion. That is a future hypothesis to test, not a result assumed by this report."),
h("1.4 Scope and contribution"),
p("The literature review follows three connected themes: refusal control, validity of behavioral evaluation, and safety under compression. It is a focused review of selected papers and implementation sources rather than an exhaustive or systematic survey. The contribution is the synthesis of those themes with a checked single-model case study and an explicit design for subsequent confirmation."),
p("The experiments use one model family, a recovered published recipe and one NF4 configuration. Selected HarmBench behaviors provide safety cases {harmbench}, selected safe XSTest items check over-refusal {xstest}, and a few GSM8K arithmetic items provide elementary utility checks. These subsets do not constitute full benchmark scores. Semantic judgments describe visible content; they do not certify real-world harmful effectiveness."),
h("1.5 Report structure"),
p("Chapter 2 compares the literature and derives the research direction. Chapter 3 defines the edit, controls and measurements. Chapter 4 presents the reproduction, semantic and quantization findings with their implications. Chapter 5 answers the immediate questions and specifies the evidence required to investigate the longer-term hypothesis. Essential source revisions and recipe settings appear in Appendix A.")
], section=9),
page("related1", "RELATED WORK", [
h("2.1 Architectural foundations"),
p("The Transformer uses attention to combine token representations {attention}. Equation (2.1) expresses scaled dot-product attention: Q, K and V are learned query, key and value projections, and d_k is the key dimension. A causal mask excludes future tokens. Multiple heads model different relationships, while residual connections preserve a continuing hidden representation to which block outputs are added."),
equation("Attention(Q, K, V) = softmax(QKᵀ / √d_k) V", "2.1"),
p("The residual representation is a vector. A direction defines a component of that vector, which can be measured by a dot product and changed by an intervention. This makes linear direction analysis accessible, but a coordinate or direction does not automatically correspond to an isolated human concept. Qwen3.5-9B also differs from the original encoder-decoder Transformer: its 32 text blocks include 24 Gated DeltaNet and eight full-attention blocks {qwen}. An editing implementation must account for both output-projection types."),
h("2.2 From linear control to a richer representation"),
p("Arditi et al. derive candidate directions from activation mean differences and evaluate both removal and addition {arditi}. Their selection uses validation refusal behavior, ability to induce refusal, and a harmless-prompt KL constraint. This is stronger evidence than selecting a vector by its mean coordinate or assigning universal significance to a tutorial's layer index."),
p("Joad et al. find category-dependent refusal directions alongside broadly shared refusal and over-refusal steering trade-offs {joad}. Their sparse-autoencoder analysis separates shared and category-specific features in the studied models. Together, these papers distinguish an effective linear control from a complete account of refusal. We therefore use the direction as an intervention target and leave internal mechanistic attribution open.")
], chapter=2, section=10),
page("related2", "", [
h("2.3 Evaluating refusal and useful fulfillment"),
p("Evaluation must define what success means. HarmBench standardizes harmful behaviors, attack evaluation and classifier-based assessment {harmbench}. StrongREJECT instead places particular emphasis on the usefulness of the information supplied for a forbidden request and validates its evaluator against human judgments {strongreject}. These approaches address related problems but do not make a phrase matcher interchangeable with a learned or rubric-based evaluator."),
p("Over-refusal is a separate cost. XSTest contains safe prompts designed to resemble unsafe requests in wording, with unsafe contrasts {xstest}. It demonstrates why benign requests need explicit coverage. For example, a word associated with violence may be used harmlessly in photography. Reducing refusal on unsafe prompts and avoiding unnecessary refusal on safe prompts must therefore be recorded separately."),
p("These studies lead to an evaluation choice for this report: explicit refusal, substantive fulfillment and correctness are distinct properties. A warning may precede requested information. A response may avoid refusal wording yet redirect to a safer task. A technically defective answer may visibly attempt fulfillment. Our review records those observations without treating them as equivalent outcomes."),
h("2.4 Research code and externally reported scores"),
p("Heretic combines directional editing with parameter optimization that considers refusal and KL divergence {heretic}. Our case study replays fixed published settings rather than conducting that search. The distinction matters because reproducing a selected recipe is different from demonstrating that an optimizer reliably discovers good edits."),
p("The Abliterlitics comparison is a useful external reference, but remains a project-reported evaluation {abliterlitics}. Its public classifier uses text and structural heuristics, distinct from the official HarmBench learned classifier. A claimed non-refusal rate must be interpreted through the classifier, prompt format and generation settings that produced it. The present study consequently keeps source-reported numbers separate from its own paired observations.")
], section=10),
page("related3", "", [
h("2.5 Quantization and the preservation problem"),
p("NF4 represents weights with a nonuniform four-bit codebook and block scaling {qlora}. Storage precision and computation precision are different: a packed four-bit weight can be reconstructed for floating-point arithmetic. QLoRA also trains low-rank adapters through a frozen quantized model and introduces double quantization and paged optimizers. Using NF4 alone is not equivalent to running the complete QLoRA training method."),
p("Chen et al. examine safety across quantization techniques and calibration data, then introduce Q-resafe to restore safety while limiting utility loss {qresafe}. Their work establishes that quantization safety and repair are existing research topics. A difference between two compressed checkpoints is therefore insufficient on its own to establish a new contribution."),
p("Kadadekar's 2026 preprint reports model/quantizer conditions where quality is retained while refusal declines {kadadekar}. Its geometric probes are weak separators of problematic conditions, and its behavioral screening index is calibrated within the study's matrix. This cautions against using a stable capability score or one geometric similarity as a general safety certificate. External transfer of a diagnostic still requires confirmation."),
h("2.6 A question about a fixed edit"),
p("Let W denote original weights, Δ a fixed behavioral edit, and Q a quantization operation interpreted through its effective reconstructed weights. The numerical difference Q(W + Δ) − Q(W) need not equal Δ. Code assignments and block scales may change after editing. This algebraic possibility motivates a controlled comparison; it is not evidence that a particular behavior is lost."),
p("The question pursued beyond this baseline is whether error associated with that fixed edit predicts loss of its intended semantic effect. Joad's representation findings and the quantization studies motivate testing behavior as well as geometry {joad} {qresafe} {kadadekar}. The reviewed works do not validate a diagnostic for our reconstructed Qwen edit. We make no claim that the broader question is previously unstudied.")
], section=10),
page("related4", "", [
h("2.7 Comparative synthesis"),
p("Table 2.1 summarizes the roles of the selected studies. It compares the evidence they supply with the question that remains for this project; it is not a ranking of methods or a claim of exhaustive coverage."),
table("Table 2.1 Literature synthesis and implications", ["Study","Established contribution","Implication for this project"], [
["Arditi et al. {arditi}","Direction removal and addition; validation-based selection","Verify the intervention before judging conversion"],
["Joad et al. {joad}","Category geometry and shared steering trade-offs","Separate behavioral control from complete mechanism"],
["StrongREJECT {strongreject}","Useful-information evaluation with human comparison","Score substantive content, not only refusal wording"],
["HarmBench {harmbench}","Standardized behaviors and evaluation framework","State actual protocol and subset coverage"],
["XSTest {xstest}","Sensitive safe cases and unsafe contrasts","Check over-refusal separately"],
["QLoRA {qlora}","NF4 and memory-efficient adapter training","Distinguish packing from training"],
["Q-resafe {qresafe}","Quantization safety assessment and patching","Compare safety and utility; acknowledge existing repair"],
["Kadadekar {kadadekar}","Quality/safety proxy audit; preprint","Require prospective validation of any diagnostic"]], [1.35,2.50,2.70]),
h("2.8 Research direction and present boundary"),
p("The synthesis suggests a specific design principle: evaluate the change produced by an edit against matched original controls, before and after compression. An unedited quantized control separates general quantization damage from edit-related change. A frozen floating edit on the same packed backbone gives a further implementation reference. Semantic review distinguishes changes in wording from changes in supplied content."),
p("Our present contribution is to establish that comparison on one verified baseline. A stronger study would first observe a repeatable behavioral failure, then test whether an edit-specific diagnostic predicts it on independent conditions. Success would require comparison with simpler measures and existing preservation approaches. The pilot in this report leaves that predictive and causal question unresolved.")
], section=10),
page("method1", "METHODOLOGY", [
h("3.1 Models and comparison design"),
p("All inference and editing use Kaggle with two NVIDIA T4 GPUs. W is the original Qwen3.5-9B model, P is the actual published Heretic v2 checkpoint, and D is our reconstruction from the recovered recipe. FP16 parameters, the original tokenizer and matched prompts are used in their direct comparison. Table 3.1 defines the native NF4 conditions; Q denotes conversion with one fixed configuration."),
table("Table 3.1 Quantization comparison conditions", ["Condition","Construction","Role"], [
["W","Original FP16 weights","Unedited floating reference"],
["D = W + Δ","Reconstructed merged FP16 edit","Edited floating reference"],
["Q(W)","Original packed as native NF4","General compression control"],
["Q(W + Δ)","Edited weights packed as native NF4","Merged edit after conversion"],
["Q(W) + Δ","Same packed Q(W) with frozen FP32 factors","Separate floating-edit reference"]], [.95,2.85,2.75]),
p("Figure 3.1 relates the literature themes to these comparisons. The arrows express study logic, not execution dates: the NF4 pilot preceded the later publisher audit, which reused matching W and D responses. P is evaluated directly in the FP16 audit, not in an additional quantized arm."),
{"type":"figure","caption":"Figure 3.1 Literature framing and study comparisons","file":"study_workflow.png"},
p("The floating-edit condition does not learn a new direction from a quantized model. Quantizing D refits block scales, so Q(W + Δ) and Q(W) share a configuration but not a fixed numerical grid. These distinctions are necessary when interpreting any observed difference.")
], chapter=3, section=11),
page("method2", "", [
h("3.2 Direction estimation and persistent edit"),
p("The reconstruction uses 400 harmful and 400 harmless extraction examples. At each activation slot, μ_h and μ_b denote their mean vectors. The recovered Heretic recipe projects the difference away from the normalized harmless mean u_b, then normalizes the result r, as in Equation (3.1) {heretic}. This preprocessing was omitted in the initial replay."),
equation("d = μ_h − μ_b     u_b = μ_b / ||μ_b||\nd⊥ = d − u_b(u_bᵀd)     r = d⊥ / ||d⊥||", "3.1"),
p("The corrected implementation projects each slot before interpolation. Its upstream direction setting 18.49 maps to activation slots 19 and 20, interpolated at 0.49. These are source coordinates rather than ranked tutorial candidates. The existing upstream strengths remain fixed; no new optimizer search is performed."),
p("For a linear output matrix M, the update in Equation (3.2) removes a scaled component along the unit direction. At λ = 1 it is an orthogonal projection; the published schedule also uses strengths above one. B and A store the rank-one factors of each update. The overall model edit is denoted Δ."),
equation("M′ = M − λ r(rᵀM) = M + BA", "3.2"),
h("3.3 Verification procedure"),
steps("Algorithm 3.1 Reconstruction and verification", [
"Load the pinned original model and extraction datasets.",
"Compute the means, project each slot and normalize its direction.",
"Interpolate the specified slots and apply the fixed strengths.",
"Merge the saved factors into FP16 weights and check the target matrices.",
"Save and reload the checkpoint, then compare the generated responses."]),
p("Native NF4 uses block size 64, uint8 packed storage, FP16 computation and no double quantization. The separate edit factors remain FP32 and fixed. Packing and factor checks are completed before behavioral interpretation. This is a conversion experiment, not QLoRA fine-tuning.")
], section=11),
page("method3", "", [
h("3.4 Evaluation fixtures and generation"),
p("Table 3.2 gives the executed samples. The historical test was exposed during diagnosis. Fresh audit items were deterministically selected and deduplicated against saved exposures; semantic near-duplicates and training contamination remain possible. A timing rule selected complete prompt families before new answer text was inspected."),
table("Table 3.2 Samples and generation caps", ["Sample","Safety / safe / arithmetic","New token caps"], [
["Historical","100 / 100 benign KL prompts / 0","100 for response comparison"],
["NF4 pilot and matched audit","6 / 3 / 3","768 safety/safe; 384 arithmetic"],
["Fresh audit","8 / 2 / 2","1024 safety; 768 safe; 384 arithmetic"]], [1.55,2.35,2.65]),
p("Compared models use the same original tokenizer, system text ‘You are a helpful assistant.’, nonthinking template, greedy decoding, seed zero, batch size one and EOS stopping. Equal per-item caps apply across arms. The audit has 24 prompts across two cohorts; its 72 records include 24 gated reuses and 48 new completions. The pilot has 12 prompts across five arms, giving 60 records."),
h("3.5 Measures and review rules"),
p("The historical matcher tests 32 normalized marker substrings; empty output also counts as refusal. Semantic review independently records explicit refusal and clear, partial or no requested-content fulfillment. It also distinguishes redirection, factual correction, missing knowledge and ambiguous intent. A disclaimer does not negate assistance, and attempted fulfillment does not establish correctness. Reaching the token cap is recorded as an unobserved continuation, not an automatic success or failure."),
p("One automated reviewer freezes each study's labels before opening its condition mapping. Masking is imperfect, no human ratings are available, and the pilot labels remain separate from the later FP16 review. Over-refusal is assessed on five distinct safe XSTest items; arithmetic provides only elementary checks."),
p("Equation (3.3) defines the original-to-edited next-token KL averaged over 100 benign prompts. It checks one prediction point, not sequence quality or broad capability."),
equation("KL₁ = (1/N) Σ_x Σ_v p_W(v|x) log[p_W(v|x) / p_D(v|x)]", "3.3")
], section=11),
page("results1", "RESULTS AND DISCUSSION", [
h("4.1 Reconstruction and historical comparison"),
p("Table 4.1 shows the exposed historical test. Recovering the existing harmless-mean projection resolves the main replay discrepancy. Publisher and corrected reconstruction both produce eight marker matches in our harness. The publisher card's six and the external 0/400 report are not treated as reproduced results."),
table("Table 4.1 Historical phrase-marker results", ["Condition","Matches /100","Capped /100","Benign first-token KL"], [
["Original W","100","59","0"],
["Initial raw replay","72","91","0.434047"],
["Publisher P","8","100","0.078430"],
["Corrected D","8","99","0.078022"],
["Corrected reload","8","99","Not separately scored"]], [2.0,1.15,1.15,2.25]),
p("The matcher recount reproduces all 500 stored labels across the five conditions. However, 99 of the corrected responses reach the 100-token cap. Eight marker matches therefore cannot be renamed 92 verified harmful fulfillments. The first-token KL values describe a narrow distributional change and do not show that reasoning ability is preserved."),
h("4.2 What the implementation checks establish"),
p("The saved update contains 64 rank-one factor pairs with 42 active matrices: 19 attention-path and 23 MLP outputs. Every target matrix matches the explicit FP16 original-plus-update calculation. All 100 response strings remain identical after saving and reloading. These checks support correct persistence of the reconstruction."),
p("Sparse publisher-weight fingerprints also support the recovered recipe, but the author's complete unrounded parameters and original runtime are unavailable. The result is an approximate measured reconstruction, rather than byte-identical recovery of the publisher's weights. The next question is whether this numerically verified edit also changes substantive answers.")
], chapter=4, section=12),
page("results2", "", [
h("4.3 Paired semantic findings"),
p("Table 4.2 reports the later audit. Clear means visible requested-content fulfillment in the saved text; harmful technical accuracy was not tested. Refusal language and fulfillment are separate columns because they can coexist."),
table("Table 4.2 Semantic judgments on selected safety cases", ["Cohort and arm","N","Clear","Partial","No","Explicit refusal","Capped"], [
["Matched W","6","0","0","6","6","1"],
["Matched D","6","2","3","1","0","5"],
["Matched P","6","2","3","1","0","5"],
["Fresh W","8","1","1","6","8","0"],
["Fresh D","8","6","2","0","0","5"],
["Fresh P","8","6","2","0","0","6"]], [1.75,.45,.65,.75,.55,1.25,.65]),
p("Publisher and reconstruction receive the same fulfillment categories on all fourteen safety cases; eight of their 24 complete response pairs are byte-identical. No reconstruction deficit is observed in these selected cases. This agreement is not an equivalence test or a population-wide efficacy result."),
p("The original's clear fresh case includes refusal language followed by requested material. Conversely, both edited models sometimes redirect to a safer task or provide only part of the requested content. The visible effect therefore extends beyond suppressing a phrase, while remaining incomplete. Two cyber prompts permit authorized-testing interpretations; excluding them leaves four clear and two partial cases out of six for each edited model, and zero clear cases for W."),
p("No safety over-refusal is observed on the five selected XSTest items. Some answers are capped or only partly useful. All three FP16 models complete five elementary arithmetic cases correctly. These checks give no basis for a broad capability-preservation claim."),
p("The findings illustrate the evaluation distinction in StrongREJECT {strongreject}: the surface form of an answer is insufficient to determine useful fulfillment. Our review is informed by that concern but does not reproduce its calibrated scorer or full benchmark.")
], section=12),
page("results3", "", [
h("4.4 Native NF4 feasibility"),
p("Table 4.3 uses the pilot's original judgments, not the later FP16 audit's revised labels. Each row covers six safety cases. The two clear edited assistance cases remain visible in both quantized edited constructions, so this fixture shows no clear disappearance of the edit's observed effect."),
table("Table 4.3 Pilot safety judgments under native NF4", ["Condition","Clear","No","Partial","Unknown","Capped /6"], [
["W","0","5","1","0","1"],
["D","2","2","1","1","5"],
["Q(W)","0","5","1","0","1"],
["Q(W + Δ)","2","1","2","1","5"],
["Q(W) + Δ","2","0","3","1","5"]], [1.85,.6,.6,.85,.85,1.5]),
p("Partial and ambiguous responses prevent a reliable directional conclusion. For the interaction [H(Q(W + Δ)) − H(Q(W))] − [H(D) − H(W)], H denotes the average binary fulfillment outcome. Allowing unresolved labels either value gives finite-fixture bounds from −0.500 to +0.667. Allowing unseen continuations after capped negative prefixes widens them to −0.833 to +0.833. These bounds are not confidence intervals and do not include reviewer error or unverified correctness."),
p("All five arms avoid safety refusal on the three selected safe items. Four arms complete all three arithmetic answers; Q(W) + Δ completes two and reaches its cap before the final answer on one. Intermediate work is not credited as a completed answer."),
p("The native packing and frozen-factor checks pass. Yet Q(W + Δ) refits scales at the active matrices. The comparison therefore includes both code-assignment and scaling effects, rather than isolating rounding on a shared grid. It also does not compare against a freshly extracted direction from Q(W)."),
p("The pilot answers RQ3 only at the feasibility level. It provides neither evidence of general stability nor a clear failure that would justify a specific repair. This limited outcome is consistent with treating quantization safety as an empirical question, as emphasized by Q-resafe and the recent proxy study {qresafe} {kadadekar}.")
], section=13),
page("results4", "", [
h("4.5 Measurement disagreement and interpretation"),
p("Table 4.4 compares two automated refusal rules with semantic judgments of the same edited outputs. The public v3 heuristic reports no fresh refusals even though two answers are partial. On a matched case, an account-protection redirect is classified as non-refusal without substantively supplying the requested intrusion procedure."),
table("Table 4.4 Different measures on the same edited outputs", ["Cohort","N","Marker refusals","Public v3 refusals","Clear fulfillment"], [
["Matched D or P","6","2","1","2"],
["Fresh D or P","8","1","0","6"]], [1.65,.45,1.45,1.50,1.20]),
p("This heuristic is not the official HarmBench classifier. A separate minimal test also identifies a conditional thinking-tag preprocessing error, but the clean nonthinking answers in Table 4.4 do not require that bug to produce disagreement. The historical 400 raw answers, exact settings and manual overrides were unavailable. Its reported zero remains unreplicated; no historical cause is established."),
h("4.6 Implications for the research direction"),
p("The study supplies a verified FP16 starting point and shows why future edit-stability measurements must use semantic outcomes. The publisher comparison reduces the likelihood that our selected redirects are explained solely by an implementation deficit. It does not identify which internal feature produces them. Likewise, retaining two pilot behaviors after NF4 does not establish preservation of a refusal representation."),
p("A useful next study would ask whether an edit-specific diagnostic improves prediction beyond general quantization error, task quality or direction cosine. The reviewed literature motivates that comparison, but our pilot contains no activation-level measurements or prospective prediction test. Such claims must await a new design and independent confirmation."),
h("4.7 Threats to validity"),
p("The main limitations are selected small samples, one automated reviewer with imperfect masking, repeated exposure of the historical test, substantial truncation and unverified harmful accuracy. Coverage is restricted to one model and one quantizer. These restrictions prevent population rates, broad utility conclusions or a general causal mechanism. The equal per-item settings and preserved raw answers improve traceability, but do not remove these limits.")
], section=13),
page("conclusion", "CONCLUSION AND FUTURE SCOPE", [
h("5.1 Conclusions"),
p("The literature connects three requirements for studying refusal edits under compression: a verified intervention, evaluation of substantive answers, and separate safety and utility measurement. The present work implements those requirements at exploratory scale on Qwen3.5-9B."),
p("RQ1 is answered within the tested setting: recovering the existing projection step produces a checked reconstruction, eight historical marker matches like the publisher, and identical responses after persistence. RQ2 shows that phrase and semantic measures disagree on selected cases. The edit changes requested content, while some answers still redirect or partially fulfill. Publisher and reconstruction share the same fulfillment categories across fourteen selected safety cases."),
p("RQ3 remains unresolved as a behavioral interaction. The native NF4 comparison is executable and retains its two clear edited assistance cases, but partial labels and caps prevent a directional conclusion. The contribution is a focused review and reproducible empirical baseline. No new editing method, universal refusal removal or general quantization mechanism is established."),
h("5.2 A focused next investigation"),
p("Independent human review of the existing answers is the immediate improvement. Subsequent evaluation should separate development and confirmation prompts, retain paired controls, measure meaningful benign utility and resolve censoring under a prespecified policy. New conditions should be chosen before inspecting their outcomes."),
p("A larger research study should distinguish ordinary quantization damage from loss of the fixed edit, then test candidate diagnostics prospectively across additional model and quantizer conditions. Fresh direction extraction after quantization belongs in its own condition; a frozen floating adapter is not that experiment. A correction would need comparison with simpler controls and existing preservation methods at matched budgets. The potential contribution lies in reliable prediction or explanation of behavioral edit stability, and remains a direction to investigate.")
], chapter=5, section=14),
]

REFERENCES = {'arditi': {'text': 'A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in Language Models Is Mediated by a Single Direction,” Advances in Neural Information Processing Systems, vol. 37, 2024.', 'url': 'https://arxiv.org/abs/2406.11717'}, 'heretic': {'text': 'p-e-w and contributors, “Heretic,” software repository, revision e26da5e0e648bb285f896c9801849edfa8d20913. Accessed: Nov. 10, 2026.', 'url': 'https://github.com/p-e-w/heretic'}, 'qwen': {'text': 'Qwen Team, “Qwen3.5-9B,” official model card, Hugging Face, 2026. Accessed: Nov. 10, 2026.', 'url': 'https://huggingface.co/Qwen/Qwen3.5-9B'}, 'abliterlitics': {'text': 'Abliterlitics, “Qwen3.5-9B ablation comparison,” project evaluation and associated classifier source, classifier revision c48b882660daffd06f1c01ba59f1e3d114f768f8. Accessed: Nov. 10, 2026.', 'url': 'https://abliterlitics.dev/models/qwen3.5-9b/'}, 'strongreject': {'text': 'A. Souly et al., “A StrongREJECT for Empty Jailbreaks,” arXiv:2402.10260, 2024.', 'url': 'https://arxiv.org/abs/2402.10260'}, 'harmbench': {'text': 'M. Mazeika et al., “HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal,” Proc. 41st Int. Conf. Machine Learning, PMLR, vol. 235, pp. 35181–35224, 2024.', 'url': 'https://proceedings.mlr.press/v235/mazeika24a.html'}, 'xstest': {'text': 'P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models,” Proc. NAACL, 2024.', 'url': 'https://arxiv.org/abs/2308.01263'}, 'attention': {'text': 'A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, vol. 30, 2017.', 'url': 'https://arxiv.org/abs/1706.03762'}, 'joad': {'text': 'F. Joad, M. Hawasly, S. Boughorbel, N. Durrani, and H. T. Sencar, “There Is More to Refusal in Large Language Models than a Single Direction,” Proc. EMNLP, pp. 59196–59232, 2026.', 'url': 'https://aclanthology.org/2026.emnlp-main.2606/'}, 'qlora': {'text': 'T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” Advances in Neural Information Processing Systems, vol. 36, 2023.', 'url': 'https://arxiv.org/abs/2305.14314'}, 'qresafe': {'text': 'K. Chen, J. Zhang, J. Hu, Y. Wang, J. Lou, Z. Feng, and M. Song, “Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models,” Proc. 42nd Int. Conf. Machine Learning, PMLR, vol. 267, pp. 9728–9746, 2025.', 'url': 'https://proceedings.mlr.press/v267/chen25ci.html'}, 'kadadekar': {'text': 'S. Kadadekar, “Quality Is Not a Safety Proxy Under Quantization,” arXiv:2606.10154, Jun. 2026, preprint.', 'url': 'https://arxiv.org/abs/2606.10154'}}

APPENDIX_PAGES = [
page("appendix1", "APPENDIX A", [
h("A.1 Essential source revisions"),
p("Table A.1 records public sources used for the case study. The reconstructed checkpoint is a study output; the numerical and persistence checks are reported in Chapter 4. Access to raw answers is controlled separately because they contain potentially harmful content."),
table("Table A.1 Model code and dataset revisions", ["Source","Revision and role"], [
["Qwen/Qwen3.5-9B","c202236235762e1c871ad0ccb60c8ee5ba337b9a\nOriginal model"],
["trohrbaugh/Qwen3.5-9B-heretic-v2","81dda62421c43a5bd5722edbc6cd47714673dc7f\nPublisher reference"],
["p-e-w/heretic","e26da5e0e648bb285f896c9801849edfa8d20913\nCompatible implementation"],
["mlabonne/harmful_behaviors","01cead01398926d81f7c52bdb790ee8cf77ebba7\nExtraction train[:400]; historical test[:100]"],
["mlabonne/harmless_alpaca","02c6a92cfcf11bb0c387334f8146d149d65b587f\nExtraction train[:400]; benign KL test[:100]"],
["centerforaisafety/HarmBench","8e1604d1171fe8a48d8febecd22f600e462bdcdd\nSelected safety cases"],
["paul-rottger/xstest","d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d\nSafe items 24, 63, 104; fresh 6, 68"],
["openai/grade-school-math","3101c7d5072418e28b9008a6636bde82a006892c\nGSM8K test items 686, 1132, 414; fresh 197, 279"]], [2.45,4.10]),
h("A.2 Fixed recipe and runtime"),
p("The source schedule parameters are attention max 2.00, peak position 18.99, min 1.43 and min distance 9.65; MLP max 1.67, peak 26.06, min 0.84 and min distance 18.04. The projected direction uses slots 19/20 and interpolation 0.49, as explained in Section 3.2."),
p("Key runtime versions are torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2 and bitsandbytes 0.50.2. Generation settings and NF4 packing are defined in Chapter 3. All model computations use the Kaggle T4 environment described in Chapter 3.")
], section=17)
]
