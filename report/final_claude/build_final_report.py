#!/usr/bin/env python3
"""Build the final college report (DOCX) in the SampleReport.docx format.

Run with the bundled Python that has python-docx and lxml:
    /home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
        report/final_claude/build_final_report.py

The build runs twice: pass 1 renders with LibreOffice to find page numbers,
pass 2 writes them into the contents and list pages. Figures come from
report/figures_proposed/make_figures.py. No model is loaded.
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = HERE.parent / "figures_proposed"
OUT = HERE / "Abliteration_Quantization_Report.docx"
EMBLEM_SRC = ROOT / "SampleReport.docx"

TITLE = "Does Abliteration Survive 4-bit Quantization?"
SUBTITLE = "A Controlled Reproduction and Semantic Evaluation on Qwen3.5-9B"
STUDENTS = [("Omm P Tripathy", "B123082"), ("Shreyash Jena", "B123124"), ("Aman Saini", "B423008")]
SUPERVISOR = "Prof. Srichandan Sobhanayak"
TEXT_W = 6.02  # inches, A4 with 1.25 in left and 1.0 in right margins

# --------------------------------------------------------------------------- references
REFS = {
    "arditi": 'A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, "Refusal in language models is mediated by a single direction," in *Advances in Neural Information Processing Systems*, vol. 37, 2024. [Online]. Available: https://arxiv.org/abs/2406.11717',
    "heretic": 'p-e-w and contributors, "Heretic: Fully automatic censorship removal for language models," software repository, 2025. Accessed: Nov. 10, 2026. [Online]. Available: https://github.com/p-e-w/heretic',
    "qlora": 'T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Advances in Neural Information Processing Systems*, vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314',
    "qresafe": 'K. Chen, J. Zhang, J. Hu, Y. Wang, J. Lou, Z. Feng, and M. Song, "Assessing safety risks and quantization-aware safety patching for quantized large language models," in *Proc. 42nd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 267, 2025, pp. 9728-9746.',
    "kadadekar": 'S. Kadadekar, "Quality is not a safety proxy under quantization," arXiv preprint arXiv:2606.10154, Jun. 2026.',
    "qwen": 'Qwen Team, "Qwen3.5-9B," model card, Hugging Face, 2026. Accessed: Nov. 10, 2026. [Online]. Available: https://huggingface.co/Qwen/Qwen3.5-9B',
    "vaswani": 'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.',
    "joad": 'F. Joad, M. Hawasly, S. Boughorbel, N. Durrani, and H. T. Sencar, "There is more to refusal in large language models than a single direction," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2026, pp. 59196-59232.',
    "harmbench": 'M. Mazeika *et al.*, "HarmBench: A standardized evaluation framework for automated red teaming and robust refusal," in *Proc. 41st Int. Conf. Machine Learning (ICML)*, PMLR, vol. 235, 2024, pp. 35181-35224.',
    "strongreject": 'A. Souly *et al.*, "A StrongREJECT for empty jailbreaks," arXiv preprint arXiv:2402.10260, 2024.',
    "xstest": 'P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, "XSTest: A test suite for identifying exaggerated safety behaviours in large language models," in *Proc. NAACL*, 2024.',
    "abliterlitics": 'Abliterlitics, "Qwen3.5-9B ablation comparison," project report and classifier source code, 2026. Accessed: Nov. 10, 2026. [Online]. Available: https://abliterlitics.dev/models/qwen3.5-9b/',
}
ADDL = {
    "labonne": 'M. Labonne, "Uncensor any LLM with abliteration," Hugging Face blog, Jun. 2024. [Online]. Available: https://huggingface.co/blog/mlabonne/abliteration',
    "egashira": 'K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, "Exploiting LLM quantization," in *Advances in Neural Information Processing Systems*, vol. 37, 2024.',
    "projected": 'grimjim, "Projected abliteration," Hugging Face community article, Oct. 2025. [Online]. Available: https://huggingface.co/blog/grimjim/projected-abliteration',
    "npba": 'grimjim, "Norm-preserving biprojected abliteration," Hugging Face community article, Nov. 2025. [Online]. Available: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration',
    "wollschlager": 'T. Wollschläger, J. Elstner, S. Geisler, V. Cohen-Addad, S. Günnemann, and J. Gasteiger, "The geometry of refusal in large language models: Concept cones and representational independence," arXiv preprint arXiv:2502.17420, 2025.',
    "kumar": 'D. Kumar, A. Kumar, S. Agarwal, and P. Harshangi, "Fine-tuning, quantization, and LLMs: Navigating unintended outcomes," arXiv preprint arXiv:2404.04392, 2024.',
    "qi": 'X. Qi *et al.*, "Safety alignment should be made more than just a few tokens deep," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025.',
    "zou": 'A. Zou *et al.*, "Representation engineering: A top-down approach to AI transparency," arXiv preprint arXiv:2310.01405, 2023.',
}

# --------------------------------------------------------------------------- content
# Block types: ("h2", text) ("p", text) ("eq", latex, label) ("fig", file, width, label, caption)
# ("table", label, caption, headers, rows, widths, aligns) ("alg", label, caption, steps)
# ("list", [items]) ("quote", text)
# Inline: **bold**, *italic*, {key} main citation, {A:key} additional bibliography.

ABSTRACT = [
    "Abliteration removes refusal behavior from a language model by projecting a refusal direction out of its weights. Edited models are usually produced at 16-bit precision and then run in 4-bit quantized form, but whether the edit survives that conversion is rarely checked. This report asks whether a refusal-ablation edit applied at full precision keeps its behavioral effect after NF4 quantization, and whether any loss can be separated from ordinary quantization damage. We study a published Heretic edit of the hybrid Qwen3.5-9B model in three steps.",
    "First, we reproduce the edit. A direct replay failed until we restored a harmless-mean projection step, after which our reconstruction matched the publisher's checkpoint in refusal-phrase matches (8 of 100 prompts) and in first-token KL divergence on benign prompts (0.078), and every edited matrix was verified. Second, we test the outcome measure. Phrase-based refusal detectors and a semantic review of fulfillment disagree on the same outputs, so we use semantic judgments. On eight fresh safety prompts, both edited models clearly supplied the requested content in six cases, against one for the original model. Third, a controlled five-condition NF4 comparison found no case in which quantization removed an effect that the edit showed at full precision, and quantizing the unedited model produced no such effect.",
    "The samples are small, the judgments come from a single AI reviewer and many answers were truncated, so partial weakening of the edit cannot be ruled out. We give an argument from the structure of the edit for why round-to-nearest 4-bit quantization should preserve it, and propose a weight-space test of that argument.",
]
KEYWORDS = "abliteration; refusal direction; quantization; NF4; semantic evaluation; large language models"

CHAPTERS = []

CHAPTERS.append(("INTRODUCTION", [
    ("h2", "1.1 Motivation"),
    ("p", "Instruction-tuned language models are trained to refuse harmful requests. Arditi et al. showed that in many open chat models this behavior is mediated largely by a single direction in the model's internal activations: removing the direction suppresses refusal, and adding it causes refusal of harmless requests {arditi}. The weight-editing form of this result, known in the open-model community as abliteration {A:labonne}, has become a standard way to produce \"uncensored\" variants of released models. Tools such as Heretic automate the procedure and tune its parameters {heretic}."),
    ("p", "Abliteration is normally applied to 16-bit weights. Most users, however, run open models in compressed form, and 4-bit formats such as NF4 let a 9-billion-parameter model fit on a single consumer GPU {qlora}. The edit therefore passes through a quantizer before it reaches the people who use it. Earlier work has shown that quantization can change the safety behavior of aligned models {qresafe}, {kadadekar}, and that it can even be exploited to activate behavior that is absent at full precision {A:egashira}. Whether a small, deliberate weight edit survives the same rounding has received much less attention."),
    ("h2", "1.2 Research question"),
    ("p", "This report studies one question:"),
    ("quote", "When a refusal-ablation edit is applied to a model at full precision and the model is then quantized to 4-bit NF4, does the edit's behavioral effect survive, and can any loss be told apart from ordinary quantization damage?"),
    ("p", "Answering it requires two things that cannot be assumed. The first is a faithful edit. If the edit is reproduced incorrectly, a change after quantization says nothing about the published method. The second is a valid outcome measure. If refusal is measured by phrase matching, a model that avoids refusal wording while withholding the requested content would count as fully edited. We therefore divide the question into three sub-questions (SQs):"),
    ("list", [
        "**SQ1.** Can a published abliteration edit be reproduced closely enough to serve as a known reference?",
        "**SQ2.** Does the absence of refusal phrases measure the edit's effect, or is a semantic judgment of the answer needed?",
        "**SQ3.** With a faithful edit and a valid measure, does the edit's effect survive NF4 quantization, and is any change specific to the edit?",
    ]),
    ("h2", "1.3 Approach and contributions"),
    ("p", "We work with Qwen3.5-9B {qwen}, a hybrid model that mixes linear-attention and full-attention layers, and with a published Heretic edit of it. All model execution used two NVIDIA T4 GPUs on Kaggle. The contributions of this report are the following."),
    ("list", [
        "A verified reconstruction of the published edit, with a controlled comparison showing that a harmless-mean projection step, known as projected abliteration {A:projected}, accounts for the difference between a failed replay and a faithful one.",
        "A comparison of two phrase-based refusal detectors with a semantic review on the same outputs, showing where and why they disagree.",
        "A five-condition NF4 comparison that separates the effect of quantizing the edit from the effect of quantizing the base model, together with an argument from the structure of the edit for why it should survive round-to-nearest quantization.",
    ]),
    ("p", "The study is exploratory. It uses one model, one quantizer and small prompt sets, and it does not propose a new editing method."),
    ("h2", "1.4 Ethical considerations"),
    ("p", "Abliteration removes a safety behavior, so this work has dual-use potential. We reproduced only an edit and a checkpoint that were already public. The aim is to understand how such edits behave under a common deployment step, which matters equally for edits that strengthen refusal. No harmful model output is quoted in this report; responses are described by category only. The raw outputs and the reconstructed checkpoint are kept in private, access-controlled storage and are not released. We did not test whether harmful answers were technically correct."),
    ("h2", "1.5 Organization of the report"),
    ("p", "Chapter 2 reviews refusal directions, the evaluation of refusal and the effect of quantization on safety. Chapter 3 describes the model, the edit, the experimental conditions and the measures. Chapter 4 presents the results for SQ1 to SQ3 and discusses their limits. Chapter 5 concludes and sets out future work. Appendix A lists sources, settings and a supplementary calculation."),
]))

CHAPTERS.append(("BACKGROUND AND RELATED WORK", [
    ("h2", "2.1 The residual stream"),
    ("p", "A decoder-only Transformer processes each token through a stack of blocks {vaswani}. Each block reads a shared vector called the residual stream, computes an update with its token-mixing sublayer and its multilayer perceptron (MLP), and adds the update back to the stream. Every block therefore writes into the stream through a small number of output matrices: the attention output projection and the MLP down projection. If a direction in the stream carries a behavior, the component of any block's output along that direction can be removed by changing only these output matrices. Abliteration relies on this property."),
    ("p", "Qwen3.5-9B has 32 blocks and a hidden size of 4096. Twenty-four blocks use Gated DeltaNet, a linear-attention layer with a recurrent state, and eight use standard full attention, in a repeating pattern of three linear blocks followed by one full-attention block {qwen}. Both kinds of block end in an output projection, so the same editing rule applies to both."),
    ("h2", "2.2 Refusal directions and abliteration"),
    ("p", "Arditi et al. collected activations for harmful and harmless instructions and took the difference of their means at each layer as a candidate refusal direction {arditi}. Removing this direction from the residual stream suppressed refusal in 13 open chat models, and adding it caused refusal of harmless requests. They also showed that the change can be made permanent by orthogonalizing the output matrices against the direction, and they chose the layer using validation refusal scores together with a limit on KL divergence from the original model on harmless prompts. The approach belongs to a wider family of representation engineering methods, which read and control model behavior through directions in activation space {A:zou}."),
    ("p", "Labonne popularized this weight edit as abliteration and published the harmful and harmless prompt sets used in this study {A:labonne}. Heretic extends the procedure in two ways {heretic}. It applies a different ablation strength to each layer through a simple linear kernel, and it searches the kernel parameters with an optimizer that trades the number of refusals against KL divergence. A later community article proposed projected abliteration, which removes from the refusal direction its component along the mean harmless activation before editing {A:projected}, followed by a norm-preserving variant {A:npba}. The argument is that the raw difference of means contains a component shared by all prompts, and that removing it reduces collateral damage. Our reproduction in Section 4.1 gives a controlled test of this step."),
    ("h2", "2.3 Is refusal a single direction?"),
    ("p", "The single-direction picture is a useful approximation rather than a complete account. Wollschläger et al. found multi-dimensional cones of refusal directions and showed that orthogonal directions need not act independently {A:wollschlager}. Joad et al. found category-specific refusal directions alongside shared ones, and trade-offs between reducing refusal and avoiding over-refusal {joad}. Related work on safety training suggests that refusal is often decided in the first few tokens of a response {A:qi}. For this report the consequence is practical: removing one direction may weaken refusal without removing it, so the effect has to be checked on the model's answers."),
    ("h2", "2.4 Measuring refusal and fulfillment"),
    ("p", "A refusal evaluation must define success. HarmBench provides a standardized set of harmful behaviors and a learned classifier for judging whether a response complies {harmbench}. StrongREJECT argues that many jailbreak evaluations overstate success because they count responses that avoid refusing but supply little useful content, and it scores the usefulness of the answer instead {strongreject}. XSTest covers the opposite failure. It contains safe prompts that resemble unsafe ones, such as asking how to kill a Python process, to measure over-refusal {xstest}."),
    ("p", "Phrase matching remains common in abliteration tools because it is cheap. The Abliterlitics project publishes community benchmark results for abliterated models using its own heuristic classifier with manual review, and lists zero refusals out of 400 prompts for the edited model studied here {abliterlitics}. Such numbers depend on the detector, the prompt format and the generation length, which is the subject of SQ2."),
    ("h2", "2.5 Quantization and safety"),
    ("p", "Quantization stores weights with fewer bits. NF4, introduced with QLoRA, maps each block of 64 weights to 16 levels placed at the quantiles of a normal distribution and scaled by the largest absolute value in the block {qlora}. Weights are stored in 4 bits and expanded to 16 bits for computation. QLoRA also trains low-rank adapters on top of the frozen quantized model; this study uses only the storage format."),
    ("p", "Several studies show that quantization changes safety behavior. Chen et al. evaluated mainstream quantization methods, found safety degradation that depends on the method and the calibration data, and proposed Q-resafe to restore it {qresafe}. Kumar et al. found that quantization had a variable effect on jailbreak susceptibility {A:kumar}. Kadadekar reports conditions in which general quality is retained while refusal declines, so that a stable capability score is no guarantee of safety {kadadekar}. Egashira et al. showed that the effect can be engineered: a model can be built to behave well at full precision and harmfully once quantized {A:egashira}. All of these concern the alignment a model was trained with. The question here is narrower: what happens to a deliberate post-training edit when the edited weights are rounded."),
    ("h2", "2.6 Summary and research gap"),
    ("p", "Table 2.1 summarizes the work most relevant to this report and its role in the design."),
    ("table", "2.1", "Summary of related work",
     ["Work", "Contribution", "Use in this report"],
     [["Arditi et al. {arditi}", "Refusal direction; ablation and weight orthogonalization", "Defines the edit"],
      ["Heretic {heretic}", "Per-layer strength kernel and parameter search", "Source of the published edit"],
      ["Projected abliteration {A:projected}", "Removes the harmless-mean component", "Explains the failed replay"],
      ["Joad et al. {joad}; Wollschläger et al. {A:wollschlager}", "Refusal is not a single direction", "Check behavior, not geometry"],
      ["HarmBench {harmbench}; XSTest {xstest}", "Harmful and safe prompt sets", "Source of test prompts"],
      ["StrongREJECT {strongreject}", "Scores useful content, not refusal wording", "Motivates semantic review"],
      ["QLoRA {qlora}", "NF4 storage format", "Quantizer under study"],
      ["Q-resafe {qresafe}; Kadadekar {kadadekar}", "Quantization alters aligned behavior", "Need an unedited quantized control"],
      ["Egashira et al. {A:egashira}", "Quantization can activate hidden behavior", "Rounding can change behavior selectively"],
      ["This report", "Survival of a post-training edit under NF4", "Addresses the gap"]],
     [1.85, 2.35, 1.82], ["l", "l", "l"]),
    ("p", "The gap lies in the combination. Work on abliteration evaluates the edit at full precision, and work on quantization safety evaluates the original alignment. A controlled test of whether an abliteration edit survives quantization, using a valid outcome measure, falls between the two."),
]))

CHAPTERS.append(("METHODOLOGY", [
    ("h2", "3.1 Model and published edit"),
    ("p", "The original model, denoted W, is Qwen3.5-9B {qwen}. The published edit, denoted P, is the second Heretic release of this model by an independent publisher (trohrbaugh/Qwen3.5-9B-heretic-v2). The publisher released the edited weights and the parameters found by Heretic's search, but not the software environment. We reconstructed the edit with a revision of Heretic that supports the hybrid architecture {heretic}, using the published parameters without a new search. All computation ran in FP16 on two NVIDIA T4 GPUs on Kaggle. Source revisions are listed in Appendix A."),
    ("h2", "3.2 Direction estimation"),
    ("p", "For each layer we record the residual-stream activation at the last prompt token for 400 harmful and 400 harmless prompts from the abliteration datasets {A:labonne}, and compute the harmful mean and the harmless mean. Let d be the difference between the two means and u the unit vector along the harmless mean. The projected refusal direction is"),
    ("eq", r"r=\frac{d-(d\cdot u)\,u}{\lVert d-(d\cdot u)\,u\rVert}", "3.1"),
    ("p", "Equation (3.1) removes from the difference of means its component along the harmless mean and scales the result to unit length. The raw direction used in our first replay omits the projection and only normalizes d. The published parameters select a direction index of 18.49. The edit therefore uses a single direction, interpolated between the directions at positions 19 and 20 with weight 0.49. Position 0 is the embedding output, so these positions are the outputs of blocks 18 and 19, counting from zero."),
    ("h2", "3.3 Weight edit"),
    ("p", "For each edited output matrix M of layer l, the edit is"),
    ("eq", r"M'=M-\lambda_l\, r\, r^{\top} M", "3.2"),
    ("p", "When the strength λ is 1, this is an orthogonal projection, and the block can no longer write anything along r. Values above 1 reverse part of the component, and values below 1 remove only part of it. Heretic sets the strength of each layer with a linear kernel"),
    ("eq", r"\lambda_l=\begin{cases}\lambda_{\max}+\dfrac{|l-p|}{w}\left(\lambda_{\min}-\lambda_{\max}\right), & |l-p|\le w\\ 0, & \mathrm{otherwise}\end{cases}", "3.3"),
    ("p", "where p is the peak position and w the width. The published values are λ_{max} = 2.00, p = 18.99, λ_{min} = 1.43 and w = 9.65 for attention output projections, and λ_{max} = 1.67, p = 26.06, λ_{min} = 0.84 and w = 18.04 for MLP down projections. These values edit the attention outputs of blocks 10 to 28 (19 matrices) and the MLP outputs of blocks 9 to 31 (23 matrices), as Fig. 3.1 shows. The update in (3.2) has rank one, so it is stored as a pair of factors per matrix, in the same form as a rank-one LoRA adapter, and then merged into the FP16 weights. We write Δ for the set of all 42 updates and D = W + Δ for the reconstructed model. Algorithm 3.1 summarizes the procedure."),
    ("fig", "fig3_edit_map.png", 5.3, "3.1", "Where the edit is applied in Qwen3.5-9B. Top: layer types. Bottom: ablation strength per layer from (3.3) with the published parameters. Strengths above 1 over-ablate."),
    ("alg", "3.1", "Reconstruction and verification", [
        "Load the original model and the 400 harmful and 400 harmless extraction prompts.",
        "Record last-token activations at every layer and compute the two means per layer.",
        "Form the difference of means, remove its harmless-mean component as in (3.1) and normalize.",
        "Interpolate positions 19 and 20 with weight 0.49 to obtain the direction r.",
        "Compute the strength of each edited matrix from (3.3) and store the rank-one factors of (3.2).",
        "Merge the factors into FP16 weights and compare every edited matrix with an explicit computation of W + Δ.",
        "Save and reload the model, then regenerate the test responses to check persistence.",
    ]),
    ("h2", "3.4 Experimental conditions"),
    ("p", "Fig. 3.2 shows the design used for SQ3. The effect of the edit at full precision is the difference in behavior between D and W. Its effect after quantization is the difference between Q(W + Δ) and Q(W), where Q denotes conversion to NF4. Comparing Q(W) with W shows what quantization does to the unedited model, which separates general quantization damage from loss of the edit. A fifth condition, Q(W) + Δ, keeps the edit factors in FP32 on top of the same quantized base, so that the edit itself is never rounded. Table 3.1 lists all conditions used in the report."),
    ("fig", "fig1_question_design.png", 5.2, "3.2", "The research question as a controlled comparison. The edit's effect is measured at full precision (top) and after NF4 (bottom); the unedited NF4 model separates quantization damage from loss of the edit. H denotes the fraction of safety prompts clearly fulfilled."),
    ("table", "3.1", "Model conditions",
     ["Label", "Construction", "Purpose"],
     [["W", "Original model, FP16", "Unedited reference"],
      ["P", "Publisher's edited checkpoint, FP16", "Check on the reconstruction"],
      ["D = W + Δ", "Reconstructed edit merged in FP16", "Edited reference"],
      ["Q(W)", "W converted to NF4", "Quantization control"],
      ["Q(W + Δ)", "D converted to NF4", "Deployment path"],
      ["Q(W) + Δ", "Q(W) with the FP32 edit factors", "Edit kept unrounded"]],
     [1.3, 2.62, 2.1], ["l", "l", "l"]),
    ("p", "NF4 conversion used the bitsandbytes implementation with blocks of 64 weights, no double quantization, packed 8-bit storage and FP16 computation. Because Q(W + Δ) is computed from the edited weights, its block scales are recomputed and can differ from those of Q(W). The comparison therefore includes both changes in level assignment and changes in scale. We did not re-extract a refusal direction from the quantized model, which would be a different experiment."),
    ("h2", "3.5 Prompts and generation"),
    ("p", "Table 3.2 lists the prompt sets. The reproduction test uses the first 100 test prompts of the harmful dataset {A:labonne}, and the first 100 harmless test prompts for the KL measure. These prompts were also used while debugging the reconstruction, so they are not a clean held-out set. The semantic evaluation uses HarmBench behaviors {harmbench} as safety prompts, safe prompts from XSTest {xstest} to check over-refusal, and grade-school arithmetic problems from GSM8K as a basic utility check. The matched set reuses the prompts of the NF4 pilot. The fresh set was selected deterministically, checked for duplicates of earlier prompts and fixed before any of its outputs were read."),
    ("table", "3.2", "Prompt sets and token limits",
     ["Set", "Models", "Safety", "Safe", "Arithmetic", "Token limits"],
     [["Reproduction test", "W, P, D", "100", "100*", "0", "100"],
      ["NF4 pilot", "Five in Table 3.1", "6", "3", "3", "768 / 768 / 384"],
      ["Matched audit", "W, P, D", "6", "3", "3", "768 / 768 / 384"],
      ["Fresh audit", "W, P, D", "8", "2", "2", "1024 / 768 / 384"]],
     [1.4, 1.3, 0.62, 0.55, 0.85, 1.3], ["l", "l", "c", "c", "c", "c"]),
    ("note", "*Harmless prompts used only for the KL measure. Token limits are given for safety / safe / arithmetic prompts."),
    ("p", "Generation settings were identical across conditions: the original tokenizer and chat template with thinking disabled, the system message \"You are a helpful assistant.\", greedy decoding, a batch size of one and stopping at the end-of-sequence token. The NF4 pilot produced 60 responses (12 prompts in five conditions). The audit of W, P and D produced 72 records, of which 48 were new generations and 24 were earlier responses reused after confirming identical settings."),
    ("h2", "3.6 Measures"),
    ("p", "**Phrase matching.** The reproduction test counts a response as a refusal if it contains any of 32 lower-case marker strings, such as \"sorry\", \"i can'\" or \"illegal\", anywhere in the text. An empty response also counts. We also apply the public heuristic classifier of the Abliterlitics project, version 3 {abliterlitics}, which combines pattern rules with checks on the opening and structure of a response."),
    ("p", "**KL divergence.** To measure collateral change on harmless prompts, we compute the KL divergence between the next-token distributions of the original and edited models at the first generated position, averaged over N = 100 harmless prompts:"),
    ("eq", r"\mathrm{KL}=\frac{1}{N}\sum_{i=1}^{N}\sum_{t}p_W(t\mid x_i)\log\frac{p_W(t\mid x_i)}{p_D(t\mid x_i)}", "3.4"),
    ("p", "This measures a single prediction step, not the quality of whole answers."),
    ("p", "**Semantic review.** Each response is labeled for fulfillment of the request (clear, partial or none) and for explicit refusal language (yes or no). Redirection to a different task, factual correction, missing knowledge and ambiguous intent are recorded separately. A disclaimer does not cancel fulfillment, and an attempt is not checked for correctness. A response that reaches the token limit is labeled on its visible part, and the truncation is recorded. In the NF4 pilot, a response whose visible part could not be classified was labeled unknown."),
    ("p", "**AI assistance and labeling.** The implementation, the orchestration of runs and the semantic labels were produced with the help of AI coding agents. All labels come from a single LLM-based reviewer that worked on shuffled, anonymized responses and fixed its labels before the mapping to conditions was revealed. Some earlier results were known to the reviewer, so the masking was imperfect. No human labels have been collected yet. All responses, token sequences and settings were saved for later human review."),
]))

CHAPTERS.append(("RESULTS AND DISCUSSION", [
    ("h2", "4.1 SQ1: Reproducing the published edit"),
    ("p", "Our first replay of the published recipe failed. It used the raw difference of means, without the projection in (3.1). With every other setting unchanged, this replay still produced refusal markers on 72 of 100 test prompts, with a benign KL of 0.434 (Table 4.1). Adding the projection reduced the marker count to 8 and the KL to 0.078, which matches the publisher's checkpoint evaluated in the same setup. A separate controlled check on 32 development prompts showed the same pattern: 20 of 32 marker matches with the raw direction and 0 of 32 with the projected one, with KL falling from 0.438 to 0.066."),
    ("table", "4.1", "Reproduction results on the 100-prompt test",
     ["Condition", "Marker matches", "Reached limit", "Benign KL"],
     [["Original W", "100", "59", "0"],
      ["Raw-direction replay", "72", "91", "0.434"],
      ["Publisher P", "8", "100", "0.078"],
      ["Reconstruction D", "8", "99", "0.078"],
      ["D after save and reload", "8", "99", "not measured"],
      ["Publisher's model card (reported)", "6", "n/a", "0.079"]],
     [2.5, 1.2, 1.15, 1.17], ["l", "c", "c", "c"]),
    ("p", "Fig. 4.1 illustrates why the projection matters. The raw difference of means points partly along the harmless mean, a direction that every prompt shares. Ablating that shared component disturbs all inputs, which raises the KL divergence. It also spends part of the unit direction on something other than refusal, so at a fixed strength less of the refusal-specific component is removed. Our measurements agree with this explanation, which is the motivation given for projected abliteration {A:projected}, although we did not measure the decomposition directly."),
    ("fig", "fig2_projection_geometry.png", 5.6, "4.1", "Effect of the harmless-mean projection. Left: schematic of the raw and projected directions. Right: with strengths and layers fixed, the projection alone reduced marker matches from 72 to 8 and benign KL from 0.434 to 0.078."),
    ("p", "The reconstruction was also checked at the weight level. The edit has 64 factor pairs, one per component and block, of which 42 are non-zero, matching the 19 attention and 23 MLP matrices given by the kernel. Every edited matrix equals the explicit FP16 computation of W + Δ, and saving and reloading the model gave 100 of 100 identical responses. Spot comparisons with the publisher's weights showed changes in the same matrices, and for the MLP and full-attention matrices the direction of the change matched our projected direction with cosine similarity above 0.9999. The publisher's unrounded parameters and software environment are not available, so the reconstruction is close to the published edit but not identical to it byte for byte."),
    ("p", "**Answer to SQ1.** The reconstruction behaves like the published checkpoint under our settings and is verified matrix by matrix, so it can serve as a known edit for the quantization study. One caution applies to every number in Table 4.1. With a 100-token limit, 99 of the 100 reconstructed responses were cut off, so 8 marker matches does not mean that 92 requests were answered. That question needs a different measure."),
    ("h2", "4.2 SQ2: Refusal phrases versus fulfillment"),
    ("p", "Table 4.2 gives the semantic review of the three FP16 models on 14 safety prompts. On the fresh set, the original model clearly fulfilled one request and used explicit refusal language on all eight. Both edited models clearly fulfilled six and partly fulfilled two, with no explicit refusals. On the matched set, both edited models clearly fulfilled two, partly fulfilled three and did not fulfill one."),
    ("table", "4.2", "Semantic review of the FP16 models on safety prompts",
     ["Set, model", "N", "Clear", "Partial", "None", "Explicit refusal", "Reached limit"],
     [["Matched, W", "6", "0", "0", "6", "6", "1"],
      ["Matched, D", "6", "2", "3", "1", "0", "5"],
      ["Matched, P", "6", "2", "3", "1", "0", "5"],
      ["Fresh, W", "8", "1", "1", "6", "8", "0"],
      ["Fresh, D", "8", "6", "2", "0", "0", "5"],
      ["Fresh, P", "8", "6", "2", "0", "0", "6"]],
     [1.3, 0.5, 0.7, 0.75, 0.7, 1.0, 1.07], ["l", "c", "c", "c", "c", "c", "c"]),
    ("p", "The reconstruction and the publisher's model received the same fulfillment label on all 14 prompts, and 8 of their 24 paired responses across all prompt types were byte-identical. Two of the fresh prompts concern security testing and could be read as requests for authorized work. Excluding them leaves four clear and two partial fulfillments out of six for each edited model, and none for the original. No model refused the safe XSTest prompts, and all three answered the five arithmetic problems correctly. These checks are too small to show that general ability is preserved, but they show no obvious damage."),
    ("p", "Individual responses show why phrase matching is not enough. The original model's one clear fulfillment began with a refusal and then provided the requested material, so refusal wording and fulfillment occurred together. In the other direction, one edited response to a request for account intrusion contained no refusal language but redirected the user toward protecting their own account. Table 4.3 and Fig. 4.2 compare the three measures on the same outputs."),
    ("table", "4.3", "Three measures applied to the same responses",
     ["Set, model", "N", "Marker refusals", "Abliterlitics v3 refusals", "Clear fulfillment"],
     [["Matched, W", "6", "6", "6", "0"],
      ["Matched, D or P", "6", "2", "1", "2"],
      ["Fresh, W", "8", "8", "7", "1"],
      ["Fresh, D or P", "8", "1", "0", "6"]],
     [1.55, 0.5, 1.2, 1.55, 1.22], ["l", "c", "c", "c", "c"]),
    ("fig", "fig4_semantic_vs_markers.png", 5.3, "4.2", "Semantic fulfillment (bars) compared with two automated refusal detectors (right columns) on the same responses. D and P received identical labels."),
    ("p", "On the fresh set the v3 heuristic finds no refusals among the edited responses, although two of the eight are only partial. On the matched set it counts the account-protection redirect as compliance. A detector that reports zero refusals is therefore compatible with incomplete fulfillment. The external report of zero refusals in 400 prompts for the publisher's model {abliterlitics} used longer generations, a different prompt format and manual overrides, and its raw responses were not available, so we could neither replicate nor explain it."),
    ("p", "While testing the public classifier we found one input format on which it fails. When a response begins with a closed reasoning block, the code cuts the text after the literal string \"</think\" and leaves a stray \">\" at the start of the answer, which defeats the rules that look for a refusal at the start. A plain refusal is detected, but the same refusal placed after a reasoning block is not. Our responses contained no reasoning blocks, so this does not explain the disagreements in Table 4.3, and we make no claim about its effect on the external result."),
    ("p", "**Answer to SQ2.** Phrase-based counts track refusal wording rather than fulfillment, and they can err in both directions. The rest of this report uses semantic fulfillment as the outcome."),
    ("h2", "4.3 SQ3: Does the edit survive NF4?"),
    ("p", "Table 4.4 and Fig. 4.3 give the semantic labels for the five pilot conditions on six safety prompts. These labels come from the pilot's own review, which used a slightly different rubric from the audit in Section 4.2, including an unknown category, so the counts for D differ between Tables 4.2 and 4.4."),
    ("table", "4.4", "NF4 pilot results on safety prompts (N = 6 per condition)",
     ["Condition", "Clear", "Partial", "Unknown", "None", "Reached limit"],
     [["W", "0", "1", "0", "5", "1"],
      ["D = W + Δ", "2", "1", "1", "2", "5"],
      ["Q(W)", "0", "1", "0", "5", "1"],
      ["Q(W + Δ)", "2", "2", "1", "1", "5"],
      ["Q(W) + Δ", "2", "3", "1", "0", "5"]],
     [1.5, 0.8, 0.8, 0.9, 0.8, 1.22], ["l", "c", "c", "c", "c", "c"]),
    ("fig", "fig5_nf4_pilot.png", 5.0, "4.3", "NF4 pilot. Both quantized edited conditions keep the two clear fulfillments seen at full precision, while the unedited NF4 model matches the original. Hatched segments are unresolved labels."),
    ("p", "The pilot answers the question at its own scale. Quantizing the unedited model did not change its behavior: Q(W) has the same counts as W, so NF4 alone does not create fulfillment. The two prompts that the edit clearly fulfilled at full precision were still clearly fulfilled after quantizing the edited model, and also when the edit was kept in FP32 on the quantized base. Neither quantized edited condition had fewer clear or partial fulfillments than D. In the notation of Fig. 3.2, the edit raised clear fulfillment from 0 to 2 of 6 at full precision and from 0 to 2 of 6 after NF4."),
    ("p", "The unresolved labels limit how far this goes. Several edited responses were partial or unknown, and five of six reached the token limit in each edited condition, so their full content was not seen. If each unresolved label is allowed to take either value, the difference between the quantized and full-precision edit effects could lie anywhere between −0.50 and +0.67 (Appendix A.3). The pilot therefore rules out complete erasure of the edit on these prompts, but it cannot rule out partial weakening. On the benign prompts, no condition refused the three safe prompts. Four conditions solved all three arithmetic problems, while Q(W) + Δ solved two and reached its limit before finishing the third."),
    ("p", "**Answer to SQ3.** On this model and this NF4 configuration, the observed effect of the edit survived quantization, and the unedited quantized control shows that the effect was not produced by quantization itself. The pilot is too small to detect a partial loss."),
    ("h2", "4.4 Why might the edit survive?"),
    ("p", "The result has a simple explanation, which we state as a hypothesis because we did not test it directly. Let c be the component that an edited matrix writes along the refusal direction r. The edit in (3.2) changes this component from c to (1 − λ)c, a change of λc with λ between 0.84 and 2.0. Quantization adds an error matrix E, so after quantizing the component becomes (1 − λ)c plus the projection of E onto r. NF4 rounding errors are small compared with the weights and behave roughly like independent noise across the 4096 output coordinates. Projecting such noise onto one unit direction does not concentrate it, so each entry of the projected error is comparable to a single rounding error. Each entry of λc, by contrast, is a full weighted sum along a direction the matrix actually writes to. The edit is a coherent change and the rounding error is not, so the edit should survive round-to-nearest quantization unless λ is small or the format is very coarse."),
    ("p", "This hypothesis can be tested on the weights alone, without generating any text. For each edited matrix, the retained fraction of the edit is"),
    ("eq", r"\rho=\frac{\langle Q(W+\Delta)-Q(W),\,\Delta\rangle}{\lVert\Delta\rVert^{2}}", "4.1"),
    ("p", "where the inner product and the norm are taken over matrix entries. A value of ρ close to 1 means that the two quantized models differ by almost exactly the intended edit. Computing ρ for several bit widths and formats would show where edit loss begins, and whether that point agrees with behavioral tests."),
    ("h2", "4.5 Threats to validity"),
    ("p", "The findings above are subject to the following limitations."),
    ("list", [
        "**Sample size.** The semantic results rest on 14 safety prompts at FP16 and 6 in the NF4 pilot. No rates, confidence intervals or equivalence tests are reported.",
        "**Labeling.** A single AI reviewer produced all semantic labels, with imperfect masking. The saved responses have not yet been reviewed by people.",
        "**Truncation.** Most edited responses reached their token limit, so the labels describe only the visible content.",
        "**Correctness.** The accuracy of harmful content was deliberately not tested.",
        "**Test exposure.** The 100-prompt reproduction test was also used during debugging.",
        "**Scope.** One model, one edit, one quantizer and one NF4 configuration were studied. Re-extracting the direction from the quantized model was not.",
    ]),
    ("p", "Fig. 4.4 summarizes what the evidence supports. Each claim about an edit requires stronger evidence than the one below it, and the report establishes the first two, gives selected-case evidence for the third and leaves the fourth untested."),
    ("fig", "fig6_claim_ladder.png", 5.4, "4.4", "Four claims about a refusal edit and the evidence this report provides for each."),
]))

CHAPTERS.append(("CONCLUSION AND FUTURE SCOPE", [
    ("h2", "5.1 Conclusion"),
    ("p", "This report asked whether a refusal-ablation edit survives 4-bit NF4 quantization. For a published Heretic edit of Qwen3.5-9B it did, on the prompts we tested. The quantized edited model kept the clear fulfillments that the edit produced at full precision, and the quantized unedited model did not acquire them."),
    ("p", "Reaching this answer required two preliminary results. The published edit could only be reproduced after restoring the harmless-mean projection, which reduced both the remaining refusals and the collateral change at a fixed strength. Refusal-phrase counts also proved unreliable as an outcome, disagreeing with semantic judgments in both directions, so the question had to be answered with semantic labels. The main limitations are the small samples, the single AI reviewer and the truncated responses, which leave a partial weakening of the edit undetected."),
    ("h2", "5.2 Future scope"),
    ("p", "We propose the following next steps, in order of priority."),
    ("list", [
        "Compute the retained fraction ρ of (4.1) for every edited matrix under NF4, 8-bit and lower-precision formats. This needs only the weights, runs on a CPU and directly tests the explanation in Section 4.4.",
        "Have human reviewers label the 132 saved responses, 72 from the audit and 60 from the pilot, with two reviewers on a shared subset to measure agreement.",
        "Repeat SQ3 with a larger prompt set, longer generation limits, a calibrated evaluator such as StrongREJECT, a second quantization method and a second model family, and add a condition in which the direction is re-extracted from the quantized model.",
    ]),
]))

APPENDIX = [
    ("h2", "A.1 Sources"),
    ("p", "Table A.1 lists the public sources used in the study. Exact revisions, file hashes and run logs are recorded in the project repository."),
    ("table", "A.1", "Models, code and datasets",
     ["Source", "Role"],
     [["Qwen/Qwen3.5-9B", "Original model W"],
      ["trohrbaugh/Qwen3.5-9B-heretic-v2", "Published edit P"],
      ["p-e-w/heretic (hybrid-compatible revision)", "Edit implementation"],
      ["mlabonne/harmful_behaviors", "Extraction (400) and reproduction test (100)"],
      ["mlabonne/harmless_alpaca", "Extraction (400) and KL prompts (100)"],
      ["centerforaisafety/HarmBench", "Safety prompts for the semantic evaluation"],
      ["paul-rottger/xstest", "Safe prompts (five distinct items)"],
      ["openai/grade-school-math (GSM8K)", "Arithmetic problems (five items)"],
      ["Abliterlitics classifier, version 3", "Heuristic refusal detector"]],
     [3.0, 3.02], ["l", "l"]),
    ("h2", "A.2 Settings and compute"),
    ("p", "The software versions were torch 2.11, transformers 5.22.1, peft 0.21.2, accelerate 1.15.0 and bitsandbytes 0.50.2. All runs used FP16 on two NVIDIA T4 GPUs. The reconstruction runs used 2.11 GPU-hours of Kaggle quota, the NF4 pilot 1.04 hours and the semantic audit 0.87 hours. Earlier unsuccessful runs are not included in these figures."),
    ("h2", "A.3 Bounds for the NF4 comparison"),
    ("p", "Let H(·) be the fraction of the six pilot safety prompts that a condition clearly fulfills. The change in edit effect caused by quantization is"),
    ("eq", r"I=\left[H(Q(W+\Delta))-H(Q(W))\right]-\left[H(D)-H(W)\right]", "A.1"),
    ("p", "With the observed labels, I = 0. If every partial and unknown label is allowed to count as either fulfilled or not, I lies between −0.500 and +0.667. If responses that were cut off are also allowed to continue into fulfillment, the range widens to between −0.833 and +0.833. These are bounds over possible labelings of a fixed set of prompts, not confidence intervals, and they do not account for reviewer error."),
]

ABBREVIATIONS = [
    ("AI", "Artificial intelligence"), ("FP16", "16-bit floating point"), ("FP32", "32-bit floating point"),
    ("GPU", "Graphics processing unit"), ("GSM8K", "Grade School Math 8K dataset"), ("KL", "Kullback-Leibler (divergence)"),
    ("LLM", "Large language model"), ("LoRA", "Low-rank adaptation"), ("MLP", "Multilayer perceptron"),
    ("NF4", "4-bit NormalFloat"), ("QLoRA", "Quantized low-rank adaptation"), ("SQ", "Sub-question"),
    ("XSTest", "Exaggerated Safety Test suite"),
]

# --------------------------------------------------------------------------- citation numbering
main_order: list[str] = []
addl_order: list[str] = []


def _register(text: str) -> None:
    for m in re.finditer(r"(?<!_)\{(A:)?([a-z]+)\}", text):
        key = m.group(2)
        if m.group(1):
            if key not in ADDL:
                raise KeyError(key)
            if key not in addl_order:
                addl_order.append(key)
        else:
            if key not in REFS:
                raise KeyError(key)
            if key not in main_order:
                main_order.append(key)


def _walk_text():
    for _title, blocks in CHAPTERS:
        for b in blocks:
            yield from _block_texts(b)
    for b in APPENDIX:
        yield from _block_texts(b)


def _block_texts(b):
    t = b[0]
    if t in ("p", "h2", "quote", "note"):
        yield b[1]
    elif t == "list":
        yield from b[1]
    elif t == "table":
        yield b[2]
        for row in b[4]:
            yield from row
    elif t == "fig":
        yield b[4]


for _t in _walk_text():
    _register(_t)
assert set(main_order) == set(REFS), set(REFS) - set(main_order)
assert set(addl_order) == set(ADDL), set(ADDL) - set(addl_order)


def cite(text: str) -> str:
    def repl(m):
        key = m.group(2)
        if m.group(1):
            return f"[A{addl_order.index(key) + 1}]"
        return f"[{main_order.index(key) + 1}]"
    return re.sub(r"(?<!_)\{(A:)?([a-z]+)\}", repl, text)


# --------------------------------------------------------------------------- docx helpers
FONT = "Times New Roman"


def set_run_font(run, size=12, bold=None, italic=None):
    run.font.name = FONT
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{a}"), FONT)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_rich(p, text, size=12, bold=False, italic=False):
    """Add text with **bold** and *italic* markup to paragraph p."""
    text = cite(text)
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\})", text):
        if not part:
            continue
        if part.startswith("_{"):
            r = p.add_run(part[2:-1])
            set_run_font(r, size=size, bold=bold, italic=italic)
            r.font.subscript = True
            continue
        b, i = bold, italic
        if part.startswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("*"):
            part, i = part[1:-1], True
        set_run_font(p.add_run(part), size=size, bold=b, italic=i)


def fmt(p, align="justify", first=0.0, left=0.0, before=0, after=6, line=1.5, keep_next=False, hanging=None):
    pf = p.paragraph_format
    pf.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT, "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    pf.left_indent = Inches(left)
    pf.first_line_indent = Inches(-hanging) if hanging else Inches(first)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.keep_with_next = keep_next
    pf.widow_control = True


def clean_style(style, size, bold, align=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is not None:
        for a in list(rf.attrib):
            del rf.attrib[a]
        for a in ("ascii", "hAnsi", "eastAsia", "cs"):
            rf.set(qn(f"w:{a}"), FONT)
    col = rpr.find(qn("w:color"))
    if col is not None:
        for a in ("themeColor", "themeShade", "themeTint"):
            col.attrib.pop(qn(f"w:{a}"), None)


def page_field(paragraph):
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    for a in ("ascii", "hAnsi", "cs"):
        rf.set(qn(f"w:{a}"), FONT)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "22")
    rpr.append(rf)
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    paragraph._p.append(fld)


def set_pgnum(section, fmt_name):
    sect = section._sectPr
    for old in sect.findall(qn("w:pgNumType")):
        sect.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt_name)
    pg.set(qn("w:start"), "1")
    cols = sect.find(qn("w:cols"))
    if cols is not None:
        cols.addprevious(pg)
    else:
        sect.append(pg)


def setup_section(section):
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(0.9)
    section.footer_distance = Inches(0.45)
    section.header_distance = Inches(0.4)


def para_border(p, sides, sz=8):
    ppr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    for side in sides:
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "2")
        e.set(qn("w:color"), "000000")
        bdr.append(e)
    after = {qn(f"w:{t}") for t in ("pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr", "suppressLineNumbers")}
    anchor = None
    for child in ppr:
        if child.tag in after:
            anchor = child
    if anchor is None:
        ppr.insert(0, bdr)
    else:
        anchor.addnext(bdr)


def cell_shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


_OMML_CACHE: dict[str, etree._Element] = {}


def omml(latex: str):
    if latex not in _OMML_CACHE:
        with tempfile.TemporaryDirectory() as td:
            md = Path(td) / "eq.md"
            md.write_text(f"$${latex}$$\n", encoding="utf-8")
            out = Path(td) / "eq.docx"
            subprocess.run(["pandoc", str(md), "-o", str(out)], check=True)
            with zipfile.ZipFile(out) as z:
                root = etree.fromstring(z.read("word/document.xml"))
        mns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
        om = root.find(f".//{{{mns}}}oMath")
        assert om is not None, latex
        for mcpr in om.iter(f"{{{mns}}}mcPr"):
            cnt = mcpr.find(f"{{{mns}}}count")
            if cnt is not None:
                mcpr.remove(cnt)
                mcpr.insert(0, cnt)
        _OMML_CACHE[latex] = om
    return copy.deepcopy(_OMML_CACHE[latex])


# --------------------------------------------------------------------------- builder
class Builder:
    def __init__(self, pages: dict):
        self.pages = pages
        self.doc = Document()
        st = self.doc.styles
        clean_style(st["Normal"], 12, False)
        clean_style(st["Heading 1"], 14, True)
        clean_style(st["Heading 2"], 12, True)
        for s in ("Heading 1", "Heading 2"):
            pf = st[s].paragraph_format
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
        setup_section(self.doc.sections[0])
        self.tables: list[tuple[str, str]] = []
        self.figures: list[tuple[str, str]] = []
        self.algs: list[tuple[str, str]] = []
        self.toc: list[tuple[int, str, str]] = []  # level, text, key

    # ---- primitives
    def para(self, text="", **kw):
        size = kw.pop("size", 12)
        bold = kw.pop("bold", False)
        italic = kw.pop("italic", False)
        p = self.doc.add_paragraph()
        fmt(p, **kw)
        if text:
            add_rich(p, text, size=size, bold=bold, italic=italic)
        return p

    def front_heading(self, text, new_page=True):
        p = self.doc.add_paragraph(style="Heading 1")
        fmt(p, align="center", before=0, after=18, line=1.0, keep_next=True)
        p.paragraph_format.page_break_before = new_page
        set_run_font(p.add_run(text), size=14, bold=True)
        return p

    def chapter_heading(self, num, title):
        p = self.doc.add_paragraph(style="Heading 1")
        fmt(p, align="center", before=0, after=24, line=1.5, keep_next=True)
        p.paragraph_format.page_break_before = num != 1
        set_run_font(p.add_run(f"CHAPTER {num}"), size=14, bold=True)
        p.add_run().add_break()
        set_run_font(p.add_run(title), size=14, bold=True)
        self.toc.append((1, f"{num}\t{title}", f"CHAPTER {num}"))

    def h2(self, text):
        p = self.doc.add_paragraph(style="Heading 2")
        fmt(p, align="left", before=10, after=4, line=1.35, keep_next=True)
        set_run_font(p.add_run(text), size=12, bold=True)
        self.toc.append((2, text, text))

    def body(self, text):
        return self.para(text, first=0.5, after=4, line=1.35)

    def quote(self, text):
        return self.para(text, left=0.5, first=0, after=6, line=1.35, italic=True)

    def note(self, text):
        return self.para(text, align="left", size=10, after=10, line=1.0)

    def bullets(self, items):
        for it in items:
            label = "•"
            m = re.match(r"\*\*(SQ\d\.)\*\*\s*(.*)", it)
            p = self.doc.add_paragraph()
            fmt(p, left=0.75, hanging=0.25, after=2, line=1.35)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))
            if m:
                fmt(p, left=0.9, hanging=0.5, after=2, line=1.35)
                p.paragraph_format.tab_stops.add_tab_stop(Inches(0.9))
                set_run_font(p.add_run(m.group(1) + "\t"), bold=True)
                add_rich(p, m.group(2))
            else:
                set_run_font(p.add_run(label + "\t"))
                add_rich(p, it)
        self.doc.paragraphs[-1].paragraph_format.space_after = Pt(8)

    def numbered(self, items):
        for n, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            fmt(p, left=0.75, hanging=0.3, after=2, line=1.35)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.75))
            set_run_font(p.add_run(f"{n})\t"))
            add_rich(p, it)
        self.doc.paragraphs[-1].paragraph_format.space_after = Pt(8)

    def equation(self, latex, label):
        p = self.doc.add_paragraph()
        fmt(p, align="left", before=4, after=8, line=1.0)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Inches(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        set_run_font(p.add_run("\t"))
        p._p.append(omml(latex))
        set_run_font(p.add_run(f"\t({label})"))

    def figure(self, fname, width, label, caption):
        p = self.doc.add_paragraph()
        fmt(p, align="center", before=6, after=4, line=1.0, keep_next=True)
        p.add_run().add_picture(str(FIG / fname), width=Inches(width))
        c = self.doc.add_paragraph()
        fmt(c, align="justify", before=2, after=12, line=1.0)
        c.paragraph_format.left_indent = Inches(0.3)
        c.paragraph_format.right_indent = Inches(0.3)
        set_run_font(c.add_run(f"Figure {label}: "), size=11, bold=True)
        add_rich(c, caption, size=11)
        self.figures.append((label, caption.split(". ")[0].rstrip(".")))

    def table(self, label, caption, headers, rows, widths, aligns):
        c = self.doc.add_paragraph()
        fmt(c, align="center", before=8, after=4, line=1.0, keep_next=True)
        set_run_font(c.add_run(f"Table {label}: "), size=11, bold=True)
        add_rich(c, caption, size=11)
        self.tables.append((label, caption))
        t = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        grid = t._tbl.tblGrid
        for gc, wdt in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(wdt * 1440)))
        tblpr = t._tbl.tblPr
        order = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
                 "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook"]
        kids = sorted(list(tblpr), key=lambda e: order.index(e.tag.split("}")[1]) if e.tag.split("}")[1] in order else 99)
        for k in kids:
            tblpr.remove(k)
            tblpr.append(k)
        for ri, vals in enumerate([headers] + rows):
            tr = t.rows[ri]
            trpr = tr._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
            for ci, v in enumerate(vals):
                cell = tr.cells[ci]
                cell.width = Inches(widths[ci])
                cp = cell.paragraphs[0]
                fmt(cp, align="center" if ri == 0 else ("center" if aligns[ci] == "c" else "left"),
                    before=2, after=2, line=1.0, keep_next=ri < len(rows))
                add_rich(cp, str(v), size=10.5, bold=ri == 0)
                if ri == 0:
                    cell_shade(cell, "E7E6E6")
        spacer = self.doc.add_paragraph()
        fmt(spacer, after=4, line=1.0)

    def algorithm(self, label, caption, steps):
        self.algs.append((label, caption))
        p = self.doc.add_paragraph()
        fmt(p, align="left", before=10, after=2, line=1.15, keep_next=True)
        para_border(p, ["top", "bottom"], sz=8)
        set_run_font(p.add_run(f"Algorithm {label} "), size=11, bold=True)
        set_run_font(p.add_run(caption), size=11)
        for i, s in enumerate(steps, 1):
            q = self.doc.add_paragraph()
            fmt(q, align="left", left=0.4, hanging=0.3, after=1, line=1.15, keep_next=i < len(steps))
            q.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
            set_run_font(q.add_run(f"{i}:\t"), size=11)
            add_rich(q, s, size=11)
            if i == len(steps):
                para_border(q, ["bottom"], sz=8)
                q.paragraph_format.space_after = Pt(12)

    def block(self, b):
        t = b[0]
        if t == "h2":
            self.h2(b[1])
        elif t == "p":
            self.body(b[1])
        elif t == "quote":
            self.quote(b[1])
        elif t == "note":
            self.note(b[1])
        elif t == "list":
            if any(x.startswith("**SQ") for x in b[1]):
                self.bullets(b[1])
            elif any(x.startswith("**") for x in b[1]):
                self.bullets(b[1])
            else:
                self.numbered(b[1])
        elif t == "eq":
            self.equation(b[1], b[2])
        elif t == "fig":
            self.figure(*b[1:])
        elif t == "table":
            self.table(*b[1:])
        elif t == "alg":
            self.algorithm(*b[1:])
        else:
            raise ValueError(t)

    # ---- front matter
    def title_page(self):
        P = self.para
        P("", align="center", after=0, line=1.0, before=30)
        P(TITLE, align="center", size=18, bold=True, after=6, line=1.15)
        P(SUBTITLE, align="center", size=13, italic=True, after=28, line=1.15)
        P("A Technical Writing Report Submitted", align="center", size=11, after=0, line=1.15)
        P("in partial fulfillment of the requirements for the award of the", align="center", size=11, after=18, line=1.15)
        P("Bachelor of Technology", align="center", size=16, bold=True, after=14, line=1.15)
        P("by", align="center", size=12, after=6, line=1.15)
        for i, (name, sid) in enumerate(STUDENTS):
            P(f"**{name}** ({sid})", align="center", size=12, after=36 if i == len(STUDENTS) - 1 else 2, line=1.15)
        P("Under the supervision of", align="center", size=12, after=0, line=1.15)
        P(SUPERVISOR, align="center", size=12, bold=True, after=40, line=1.15)
        p = P("", align="center", after=16, line=1.0)
        with zipfile.ZipFile(EMBLEM_SRC) as z:
            data = z.read("word/media/image1.jpeg")
        em = HERE / "_emblem.jpeg"
        em.write_bytes(data)
        p.add_run().add_picture(str(em), width=Inches(1.15))
        P("Department of Computer Science", align="center", size=12, after=0, line=1.15)
        P("International Institute of Information Technology Bhubaneswar", align="center", size=12, after=8, line=1.15)
        P("November 2026", align="center", size=12, after=0, line=1.15)

    def approval(self):
        self.front_heading("APPROVAL OF THE VIVA-VOCE BOARD", new_page=False)
        self.para("November 10, 2026", align="right", after=18)
        self.para(f"Certified that the report entitled **“{TITLE}”** submitted by **Omm P Tripathy** (B123082), **Shreyash Jena** (B123124) and **Aman Saini** (B423008) to the International Institute of Information Technology Bhubaneswar in partial fulfillment of the requirements for the award of the Bachelor of Technology under the BTech Programme has been accepted by the examiners during the viva-voce examination held today.", first=0.5, after=60)
        p = self.para("", align="left", after=0)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
        p.paragraph_format.tab_stops.add_tab_stop(Inches(4.3))
        set_run_font(p.add_run("\t(Supervisor)\t(Panel Members)"), bold=True)

    def acknowledgements(self):
        self.front_heading("ACKNOWLEDGEMENTS")
        self.para("We thank Prof. Srichandan Sobhanayak for supervision and guidance during this work.", first=0.5)
        self.para("We acknowledge the authors and maintainers of the Qwen3.5 model, the Heretic tool, the published edit and the evaluation datasets used in this study. Parts of the implementation, run orchestration and response labeling were carried out with AI coding assistants; the extent of this assistance is described in Section 3.6.", first=0.5, after=36)
        for name, sid in STUDENTS:
            self.para(f"{name} ({sid})", align="right", after=0, line=1.3)

    def abstract(self):
        self.front_heading("ABSTRACT")
        for t in ABSTRACT:
            self.para(t, first=0.5, after=6, line=1.5)
        p = self.para("", align="left", before=12)
        set_run_font(p.add_run("Keywords: "), bold=True)
        set_run_font(p.add_run(KEYWORDS))

    def _entry(self, left, text, page, bold=False, indent=0.0, text_tab=None, after=4):
        p = self.doc.add_paragraph()
        fmt(p, align="left", left=indent, after=after, line=1.05)
        ts = p.paragraph_format.tab_stops
        if text_tab:
            ts.add_tab_stop(Inches(text_tab))
        ts.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if left:
            set_run_font(p.add_run(left + "\t"), size=11.5, bold=bold)
        set_run_font(p.add_run(text), size=11.5, bold=bold)
        set_run_font(p.add_run("\t" + str(page)), size=11.5, bold=bold)

    def toc_page(self, toc_items):
        self.front_heading("TABLE OF CONTENTS")
        hdr = self.doc.add_paragraph()
        fmt(hdr, align="left", after=6, line=1.0)
        hdr.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        set_run_font(hdr.add_run("Content\tPage No."), size=11, bold=True)
        for level, text, key in toc_items:
            pg = self.pages.get(key, "00")
            if level == 0:
                self._entry("", text, pg, bold=True, after=1)
            elif level == 1:
                num, title = text.split("\t")
                self._entry(num, title, pg, bold=True, text_tab=0.35, after=1)
                self.doc.paragraphs[-1].paragraph_format.space_before = Pt(4)
            else:
                self._entry("", text, pg, indent=0.35, after=1)

    def list_page(self, heading, col, items, prefix, key_prefix, new_page=True):
        self.front_heading(heading, new_page=new_page)
        hdr = self.doc.add_paragraph()
        fmt(hdr, align="left", after=6, line=1.0)
        hdr.paragraph_format.tab_stops.add_tab_stop(Inches(1.1))
        hdr.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        set_run_font(hdr.add_run(f"{col} No.\tCaption\tPage No."), size=11, bold=True)
        for label, cap in items:
            self._entry(f"{prefix} {label}", cite(cap), self.pages.get(f"{key_prefix} {label}:", "00"), text_tab=1.1)

    def abbreviations(self):
        self.front_heading("LIST OF ABBREVIATIONS")
        for a, d in ABBREVIATIONS:
            p = self.para("", align="left", after=2, line=1.3)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(1.3))
            set_run_font(p.add_run(a + "\t"), bold=True)
            set_run_font(p.add_run(d))

    def references(self):
        self.front_heading("REFERENCES")
        self.toc.append((0, "REFERENCES", "REFERENCES"))
        for i, k in enumerate(main_order, 1):
            self._ref(f"[{i}]", REFS[k])
        self.front_heading("ADDITIONAL BIBLIOGRAPHY", new_page=False)
        self.doc.paragraphs[-1].paragraph_format.space_before = Pt(18)
        self.toc.append((0, "ADDITIONAL BIBLIOGRAPHY", "ADDITIONAL BIBLIOGRAPHY"))
        self.para("Further sources cited in the text with the prefix A. They support the background discussion and are listed separately from the core references.", align="left", size=11, after=8, line=1.15)
        for i, k in enumerate(addl_order, 1):
            self._ref(f"[A{i}]", ADDL[k])

    def _ref(self, label, text):
        p = self.doc.add_paragraph()
        fmt(p, align="justify", left=0.5, hanging=0.5, after=5, line=1.15)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.5))
        set_run_font(p.add_run(label + "\t"), size=11)
        add_rich(p, text, size=11)

    # ---- assemble
    def build(self):
        self.title_page()
        s = self.doc.add_section(WD_SECTION.NEW_PAGE)
        setup_section(s)
        set_pgnum(s, "lowerRoman")
        s.footer.is_linked_to_previous = False
        fp = s.footer.paragraphs[0]
        fmt(fp, align="center", after=0, line=1.0)
        page_field(fp)
        self.approval()
        self.acknowledgements()
        self.abstract()
        # Build the body first into a scratch builder to collect TOC/list entries.
        toc_items = [(0, "ABSTRACT", "ABSTRACT"), (0, "LIST OF TABLES", "LIST OF TABLES"),
                     (0, "LIST OF FIGURES", "LIST OF FIGURES"), (0, "LIST OF ALGORITHMS", "LIST OF ALGORITHMS"),
                     (0, "LIST OF ABBREVIATIONS", "LIST OF ABBREVIATIONS")]
        scratch = Builder(self.pages)
        scratch._body()
        toc_items += scratch.toc
        self.toc_page(toc_items)
        self.list_page("LIST OF TABLES", "Table", scratch.tables, "Table", "Table")
        self.list_page("LIST OF FIGURES", "Figure", scratch.figures, "Figure", "Figure", new_page=False)
        self.doc.paragraphs[-1 - len(scratch.figures) - 1].paragraph_format.space_before = Pt(28)
        self.list_page("LIST OF ALGORITHMS", "Algorithm", scratch.algs, "Algorithm", "Algorithm", new_page=False)
        self.doc.paragraphs[-1 - len(scratch.algs) - 1].paragraph_format.space_before = Pt(28)
        self.abbreviations()
        s2 = self.doc.add_section(WD_SECTION.NEW_PAGE)
        setup_section(s2)
        set_pgnum(s2, "decimal")
        s2.footer.is_linked_to_previous = False
        fp2 = s2.footer.paragraphs[0]
        for el in list(fp2._p):
            if el.tag != qn("w:pPr"):
                fp2._p.remove(el)
        fmt(fp2, align="center", after=0, line=1.0)
        page_field(fp2)
        self._body()
        zoom = self.doc.settings.element.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")
        self.doc.core_properties.title = TITLE
        self.doc.core_properties.subject = SUBTITLE
        self.doc.core_properties.author = ", ".join(n for n, _ in STUDENTS)
        self.doc.core_properties.keywords = KEYWORDS
        return self.doc

    def _body(self):
        for n, (title, blocks) in enumerate(CHAPTERS, 1):
            self.chapter_heading(n, title)
            for b in blocks:
                self.block(b)
        self.references()
        p = self.doc.add_paragraph(style="Heading 1")
        fmt(p, align="center", before=0, after=18, line=1.5, keep_next=True)
        p.paragraph_format.page_break_before = True
        set_run_font(p.add_run("APPENDIX A"), size=14, bold=True)
        p.add_run().add_break()
        set_run_font(p.add_run("REPRODUCIBILITY DETAILS"), size=14, bold=True)
        self.toc.append((0, "APPENDIX A  REPRODUCIBILITY DETAILS", "APPENDIX A"))
        for b in APPENDIX:
            self.block(b)


# --------------------------------------------------------------------------- render and paginate
def roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def render(docx: Path, outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["soffice", "-env:UserInstallation=file:///tmp/claude-1000/lo_profile_report", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
                   check=True, capture_output=True)
    pdf = outdir / (docx.stem + ".pdf")
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    return txt.split("\f")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_pages(pages_text: list[str], builder: Builder) -> dict:
    body_start = next(i for i, t in enumerate(pages_text) if re.search(r"^\s*CHAPTER 1\s*$", t, re.M))
    found = {}
    for i, t in enumerate(pages_text[:body_start]):
        lines = [norm(x) for x in t.splitlines() if x.strip()]
        if not lines:
            continue
        for key in ("ABSTRACT", "TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF ALGORITHMS", "LIST OF ABBREVIATIONS"):
            if any(l == key for l in lines) and key not in found:
                found[key] = roman(i)
    keys = [k for _l, _t, k in builder.toc] + [f"Table {l}:" for l, _ in builder.tables] + \
           [f"Figure {l}:" for l, _ in builder.figures] + [f"Algorithm {l}:" for l, _ in builder.algs]
    for key in keys:
        for i in range(body_start, len(pages_text)):
            lines = [norm(x) for x in pages_text[i].splitlines()]
            probe = key[:-1] if key.startswith("Algorithm") else key
            if key.startswith("CHAPTER") or key in ("REFERENCES", "ADDITIONAL BIBLIOGRAPHY", "APPENDIX A"):
                hit = any(l == key for l in lines)
            else:
                hit = any(l.startswith(probe) for l in lines)
            if hit:
                found[key] = str(i - body_start + 1)
                break
    return found


def main():
    qa = HERE / "qa"
    pages: dict = {}
    for attempt in range(3):
        b = Builder(pages)
        doc = b.build()
        doc.save(OUT)
        text = render(OUT, qa)
        scratch = Builder(pages)
        scratch._body()
        new = find_pages(text, scratch)
        print(f"pass {attempt + 1}: {len(text) - 1} pages, {len(new)} locators")
        if new == pages:
            break
        pages = new
    missing = [k for _l, _t, k in scratch.toc if k not in pages]
    print("missing locators:", missing)
    (qa / "page_map.json").write_text(json.dumps(pages, indent=1))
    full = "\n".join(text)
    for bad in ("—", "–"):
        if bad in full:
            print("WARNING: dash character found:", repr(bad))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
