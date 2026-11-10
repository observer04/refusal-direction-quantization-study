#!/usr/bin/env python3
"""Build the lighter-math variant of the final report.

Same findings, tables, figures and structure as build_final_report.py, but
every numbered equation is replaced by a plain-language description. Run with
the bundled Python:
    /home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 \
        report/final_claude/build_light_report.py
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("base", HERE / "build_final_report.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

LIGHT = HERE / "light"
LIGHT.mkdir(exist_ok=True)


def chapter(title):
    for t, blocks in base.CHAPTERS:
        if t == title:
            return blocks
    raise KeyError(title)


def find(blocks, kind, prefix):
    for i, b in enumerate(blocks):
        if b[0] == kind and isinstance(b[1], str) and b[1].startswith(prefix):
            return i
    raise KeyError(prefix)


def replace_span(blocks, start_prefix, end_prefix, new_blocks, start_kind="p", end_kind="p"):
    i = find(blocks, start_kind, start_prefix)
    j = find(blocks, end_kind, end_prefix)
    blocks[i:j + 1] = new_blocks


def retext(blocks, old, new):
    hits = 0
    for k, b in enumerate(blocks):
        if b[0] in ("p", "note", "quote") and old in b[1]:
            blocks[k] = (b[0], b[1].replace(old, new))
            hits += 1
        elif b[0] == "fig" and old in b[4]:
            blocks[k] = b[:4] + (b[4].replace(old, new),)
            hits += 1
        elif b[0] in ("list",) and any(old in x for x in b[1]):
            blocks[k] = (b[0], [x.replace(old, new) for x in b[1]])
            hits += 1
        elif b[0] == "alg" and any(old in x for x in b[3]):
            blocks[k] = b[:3] + ([x.replace(old, new) for x in b[3]],)
            hits += 1
    assert hits, old


# ---------------------------------------------------------------- Chapter 3
m = chapter("METHODOLOGY")
replace_span(m, "For each layer we record", "Equation (3.1) removes", [
    ("p", "For each layer we record the residual-stream activation at the last prompt token for 400 harmful and 400 harmless prompts from the abliteration datasets {A:labonne}, and compute the average activation of each group: the harmful mean and the harmless mean. The raw refusal direction is the difference between the two means. Following the published recipe, we then remove from this difference the part that points along the harmless mean, because that part is shared by all prompts rather than specific to refusal, and scale the result to unit length. The result is the refusal direction r. Fig. 4.1 in Chapter 4 sketches this step."),
    ("p", "Our first replay omitted this projection and used the raw difference directly. The published parameters select a direction index of 18.49, so the edit uses a single direction interpolated between the directions at positions 19 and 20 with weight 0.49. Position 0 is the embedding output, so these positions are the outputs of blocks 18 and 19, counting from zero."),
])
replace_span(m, "For each edited output matrix M", "where p is the peak position", [
    ("p", "The edit changes each selected output matrix so that its layer writes less along the refusal direction. For any input, the amount the matrix writes along r is measured, and a multiple λ of that amount is subtracted, where λ is the ablation strength of the layer. When λ is 1 the layer can no longer write anything along r. Values above 1 reverse part of the component, and values below 1 remove only part of it."),
    ("p", "Heretic gives each layer its own strength. The strength is highest at a chosen peak layer, falls in a straight line with distance from the peak, and is zero beyond a set width. The published settings place the attention peak near layer 19, with strengths from 2.00 down to 1.43, and the MLP peak near layer 26, with strengths from 1.67 down to 0.84; Appendix A.2 lists the exact values. As a result the edit changes the attention outputs of blocks 10 to 28 (19 matrices) and the MLP outputs of blocks 9 to 31 (23 matrices), as Fig. 3.1 shows. Each change has rank one, so it is stored as a pair of vectors per matrix, in the same form as a rank-one LoRA adapter, and then merged into the FP16 weights. We write Δ for the set of all 42 updates and D = W + Δ for the reconstructed model. Algorithm 3.1 summarizes the procedure."),
])
retext(m, "ablation strength per layer from (3.3) with the published parameters", "ablation strength per layer from the published parameters")
retext(m, "remove its harmless-mean component as in (3.1) and normalize", "remove its harmless-mean component and normalize")
retext(m, "Compute the strength of each edited matrix from (3.3) and store the rank-one factors of (3.2).", "Compute the strength of each edited matrix and store its rank-one update factors.")
replace_span(m, "**KL divergence.**", "This measures a single prediction step", [
    ("p", "**KL divergence.** To measure collateral change on harmless prompts we use the Kullback-Leibler (KL) divergence, a standard measure of how different two probability distributions are. For each of 100 harmless prompts we compare the original and edited models' probabilities for the first token of the answer, and average the result. A value of 0 means no change, and larger values mean larger change. This measures a single prediction step, not the quality of whole answers."),
])
assert not any(b[0] == "eq" for b in m)

# ---------------------------------------------------------------- Chapter 4
r = chapter("RESULTS AND DISCUSSION")
retext(r, "without the projection in (3.1)", "without the harmless-mean projection described in Section 3.2")
replace_span(r, "The result has a simple explanation", "where the inner product and the norm", [
    ("p", "The result has a simple explanation, which we state as a hypothesis because we did not test it directly. The edit is one coordinated change: in every edited matrix it removes a large, consistent amount along a single direction, with strengths between 0.84 and 2.0. Rounding to 4 bits adds small errors to individual weights, and these errors behave roughly like random noise spread over all 4096 output directions. Along the one refusal direction such scattered errors largely cancel and stay small, while the edit is present in full. The edit should therefore survive round-to-nearest quantization unless its strength is small or the format is very coarse."),
    ("p", "This hypothesis can be tested on the weights alone, without generating any text. For each edited matrix, one can compare the actual difference between the two quantized models, Q(W + Δ) minus Q(W), with the intended edit Δ, and compute what fraction of the intended change is still present. A retained fraction close to 1 means that the quantized models differ by almost exactly the intended edit. Computing it for several bit widths and formats would show where edit loss begins, and whether that point agrees with behavioral tests."),
])
assert not any(b[0] == "eq" for b in r)

# ---------------------------------------------------------------- Chapter 5
c = chapter("CONCLUSION AND FUTURE SCOPE")
retext(c, "Compute the retained fraction ρ of (4.1) for every edited matrix", "Compute the retained fraction of the edit described in Section 4.4 for every edited matrix")

# ---------------------------------------------------------------- Appendix
a = base.APPENDIX
retext(a, "The software versions were", "The published Heretic parameters are a direction index of 18.49; for attention output projections a maximum strength of 2.00 at position 18.99, falling to 1.43 at a distance of 9.65; and for MLP down projections a maximum strength of 1.67 at position 26.06, falling to 0.84 at a distance of 18.04. The software versions were")
replace_span(a, "Let H(·) be the fraction", "With the observed labels", [
    ("p", "For each condition, let its score be the fraction of the six pilot safety prompts that it clearly fulfills. The edit effect at full precision is the score of D minus the score of W, and the edit effect after NF4 is the score of Q(W + Δ) minus the score of Q(W). The change caused by quantization is the second effect minus the first. With the observed labels this change is 0, since both effects are 2 of 6. If every partial and unknown label is allowed to count as either fulfilled or not, the change lies between −0.500 and +0.667. If responses that were cut off are also allowed to continue into fulfillment, the range widens to between −0.833 and +0.833. These are bounds over possible labelings of a fixed set of prompts, not confidence intervals, and they do not account for reviewer error."),
])
assert not any(b[0] == "eq" for b in a)

# No leftover references to numbered equations.
import re
for _t, blocks in base.CHAPTERS + [("A", base.APPENDIX)]:
    for b in blocks:
        for txt in base._block_texts(b):
            assert not re.search(r"\((3\.[1-4]|4\.1|A\.1)\)", txt), txt

base.HERE = LIGHT
base.OUT = LIGHT / "Abliteration_Quantization_Report_LightMath.docx"
base.main()
(LIGHT / "_emblem.jpeg").unlink(missing_ok=True)
