"""Build the presentation deck for the abliteration and NF4 report.

Run with the bundled Python (python-pptx 1.0.2):
    python3.12 build_slides.py
Writes Abliteration_Quantization_Slides.pptx next to this script.
All numbers are copied from report/final_claude/build_final_report.py.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures_proposed"
ASSETS = HERE / "assets"
OUT = HERE / "Abliteration_Quantization_Slides.pptx"

# --------------------------------------------------------------------------- style
HEAD = "Cambria"
BODY = "Calibri"
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
ACCENT_DK = RGBColor(0x1B, 0x4F, 0x9C)
TINT = RGBColor(0xEA, 0xF2, 0xFC)
TINT2 = RGBColor(0xD3, 0xE4, 0xF8)
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x5B, 0x65, 0x73)
GREY = RGBColor(0x9A, 0xA3, 0xAE)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SHORT_TITLE = "Does Abliteration Survive 4-bit Quantization?"
TOTAL_MAIN = 19

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# --------------------------------------------------------------------------- figure prep
def prep_figures() -> dict[str, Path]:
    ASSETS.mkdir(exist_ok=True)
    out = {}
    # fig1: drop the formula header so the slide carries the words instead
    im = Image.open(FIG / "fig1_question_design.png").convert("RGB")
    im = im.crop((0, 155, im.width, im.height))
    p = ASSETS / "fig1_design_noformula.png"
    im.save(p, dpi=(300, 300))
    out["fig1"] = p
    # fig2: blank out the formula line under the schematic (left panel only)
    im = Image.open(FIG / "fig2_projection_geometry.png").convert("RGB")
    ImageDraw.Draw(im).rectangle((0, 815, 1150, im.height), fill=(255, 255, 255))
    p = ASSETS / "fig2_projection_noformula.png"
    im.save(p, dpi=(300, 300))
    out["fig2"] = p
    for k in ("fig3_edit_map", "fig4_semantic_vs_markers", "fig5_nf4_pilot"):
        out[k.split("_")[0]] = FIG / f"{k}.png"
    return out


# --------------------------------------------------------------------------- helpers
def _rgb(c):
    return c if isinstance(c, RGBColor) else RGBColor.from_string(c)


def set_bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _apply_run(run, size, bold, color, italic, font):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _rgb(color)
    run.font.name = font


def _bullet(p, level_in=0.32, color=ACCENT):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(level_in)))
    pPr.set("indent", str(-Inches(level_in)))
    buClr = etree.SubElement(pPr, qn("a:buClr"))
    clr = etree.SubElement(buClr, qn("a:srgbClr"))
    clr.set("val", str(color))
    buFont = etree.SubElement(pPr, qn("a:buFont"))
    buFont.set("typeface", "Arial")
    buChar = etree.SubElement(pPr, qn("a:buChar"))
    buChar.set("char", "•")


def text(slide, x, y, w, h, paras, size=18, color=INK, bold=False, italic=False,
         font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=6,
         margin=0.0, name=None, line_spacing=None):
    """paras: list of paragraphs. Each is a str, or a dict with keys
    t (str or list of (str, opts) runs), size, bold, color, italic, bullet, align,
    space_after, font."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    m = Inches(margin)
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = m
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        if isinstance(para, str):
            para = {"t": para}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = para.get("align", align)
        p.space_after = Pt(para.get("space_after", space_after))
        if line_spacing:
            p.line_spacing = line_spacing
        runs = para["t"]
        if isinstance(runs, str):
            runs = [(runs, {})]
        for rt, ro in runs:
            r = p.add_run()
            r.text = rt
            _apply_run(
                r,
                ro.get("size", para.get("size", size)),
                ro.get("bold", para.get("bold", bold)),
                ro.get("color", para.get("color", color)),
                ro.get("italic", para.get("italic", italic)),
                ro.get("font", para.get("font", font)),
            )
        if para.get("bullet"):
            _bullet(p, color=para.get("bullet_color", ACCENT))
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        line_w=1.25, dash=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
        if dash:
            s.line.dash_style = dash
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    _drop_style(s)
    s.text_frame.text = ""
    return s


def _drop_style(shape):
    """Remove the theme style reference so no inherited shadow or outline is drawn."""
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def circle_num(slide, x, y, d, label, fill=ACCENT, color=WHITE, size=20):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _apply_run(r, size, True, color, False, BODY)
    return c


def line(slide, x1, y1, x2, y2, color=INK, width=2.0, head=False, dash=None, head_size="med"):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _drop_style(c)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle")
        tail.set("w", head_size)
        tail.set("len", head_size)
    return c


def picture(slide, path, x, y, w=None, h=None, alt=""):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def chevron(slide, x, y, w=0.45, h=0.5, color=ACCENT):
    s = box(slide, x, y, w, h, fill=color, shape=MSO_SHAPE.CHEVRON)
    return s


def notes(slide, presenter, body, qa=()):
    parts = [f"Presenter: {presenter}", "", body]
    if qa:
        parts += ["", "Likely questions:"]
        for q, a in qa:
            parts += [f"Q: {q}", f"A: {a}"]
    slide.notes_slide.notes_text_frame.text = "\n".join(parts)
    slide._note_body = body  # for the word-count check


def content_slide(kicker, title, num):
    s = prs.slides.add_slide(BLANK)
    set_bg(s)
    if kicker:
        text(s, 0.6, 0.38, 12.1, 0.38, kicker.upper(), size=15, bold=True, color=ACCENT, name="Kicker")
    text(s, 0.6, 0.74, 12.1, 0.85, title, size=30, bold=True, font=HEAD, color=INK,
         anchor=MSO_ANCHOR.TOP, name="Title")
    text(s, 0.6, 7.02, 8.0, 0.3, SHORT_TITLE, size=11, color=MUTED, name="Footer")
    text(s, 11.73, 7.02, 1.0, 0.3, str(num), size=11, color=MUTED, align=PP_ALIGN.RIGHT, name="Slide number")
    return s


SLIDES = []


def add(fn):
    SLIDES.append(fn)
    return fn


def layer_strip(slide, x, y, w, h=0.32, gap=0.05):
    """32 layers of Qwen3.5-9B; edited blocks (9 to 31) in blue."""
    cw = (w - gap * 31) / 32
    for i in range(32):
        full = i % 4 == 3
        edited = 9 <= i <= 31
        if edited:
            fill = ACCENT_DK if full else ACCENT
        else:
            fill = INK if full else RGBColor(0xD6, 0xD6, 0xD1)
        box(slide, x + i * (cw + gap), y, cw, h, fill=fill, shape=MSO_SHAPE.RECTANGLE)


# --------------------------------------------------------------------------- slides
@add
def s01_title(F):
    s = prs.slides.add_slide(BLANK)
    set_bg(s)
    text(s, 0.8, 0.6, 11.7, 0.45, "B.Tech Technical Writing Report  |  Presentation", size=16, color=ACCENT, bold=True)
    text(s, 0.8, 1.05, 11.0, 1.45, "Does Abliteration Survive\n4-bit Quantization?", size=40, bold=True, font=HEAD)
    text(s, 0.8, 2.6, 11.7, 0.6, "A Controlled Reproduction and Semantic Evaluation on Qwen3.5-9B",
         size=24, color=ACCENT_DK, font=HEAD)
    layer_strip(s, 0.8, 3.45, 7.2)
    text(s, 8.2, 3.37, 4.6, 0.5, "the 32 layers of Qwen3.5-9B; blue layers carry the edit", size=15, color=MUTED,
         italic=True)
    names = [("Omm P Tripathy", "B123082"), ("Shreyash Jena", "B123124"), ("Aman Saini", "B423008")]
    for i, (n, r) in enumerate(names):
        text(s, 0.8 + i * 3.6, 4.25, 3.4, 0.9,
             [{"t": n, "size": 21, "bold": True, "space_after": 2}, {"t": r, "size": 17, "color": MUTED}])
    text(s, 0.8, 5.3, 11.7, 1.6, [
        {"t": [("Supervisor: ", {"bold": True}), ("Prof. Srichandan Sobhanayak", {})], "size": 18},
        {"t": "Department of Computer Science", "size": 17, "color": MUTED, "space_after": 2},
        {"t": "International Institute of Information Technology Bhubaneswar", "size": 17, "color": MUTED},
        {"t": "November 2026", "size": 17, "color": MUTED},
    ])
    notes(s, "Omm",
          "Good morning. We are Omm, Shreyash and Aman, and our supervisor is Prof. Srichandan Sobhanayak. "
          "Our report asks a simple practical question. If someone removes the safety refusals from a language "
          "model and then compresses that model to 4 bits so it runs on a small GPU, is the removal still there "
          "afterwards? I will cover the motivation and the background ideas. Shreyash will explain how we rebuilt "
          "the edit and how we measured its effect. Aman will present the quantization result, the limitations "
          "and our conclusions.",
          [("How is the talk split?", "Omm: slides 1 to 6. Shreyash: slides 7 to 12. Aman: slides 13 to 19. "
            "The backup slide is for questions on the reproduction numbers.")])
    return s


@add
def s02_why(F):
    s = content_slide("Motivation", "Edited models get shrunk, but the edit is rarely checked", 2)
    steps = [
        ("Chat models refuse", "They are trained to refuse harmful requests."),
        ("Abliteration", "Deletes one “refusal direction” from the weights, so the model stops refusing."),
        ("Shrink to 4-bit", "Users compress the model so a 9-billion-parameter model fits on one consumer GPU."),
        ("Still edited?", "Whether the edit survives the shrinking is rarely checked."),
    ]
    x0, w, gap, y, h = 0.6, 2.75, 0.37, 2.15, 3.0
    for i, (head, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        last = i == len(steps) - 1
        box(s, x, y, w, h, fill=TINT2 if last else TINT, line=ACCENT if last else None, line_w=2.0)
        circle_num(s, x + 0.25, y + 0.25, 0.55, "?" if last else str(i + 1), size=20)
        text(s, x + 0.25, y + 0.95, w - 0.45, 0.5, head, size=20, bold=True, color=ACCENT_DK if last else INK)
        text(s, x + 0.25, y + 1.5, w - 0.45, 1.45, body, size=17)
        if not last:
            chevron(s, x + w + 0.07, y + h / 2 - 0.22, 0.24, 0.44)
    text(s, 0.6, 5.55, 12.1, 1.1, [
        {"t": "Earlier work studied how quantization changes a model’s trained safety. "
              "We ask a narrower question: does a small, deliberate weight edit survive the same rounding?",
         "size": 18, "color": MUTED},
    ])
    notes(s, "Omm",
          "Chat models are trained to refuse harmful requests. Researchers found that in many models this refusal "
          "is controlled largely by a single direction inside the model's internal activations. Abliteration is a "
          "weight edit that deletes that direction, and tools such as Heretic automate it. These edits are made on "
          "16-bit weights. But most people run open models in 4-bit form, because a 9-billion-parameter model then "
          "fits on one consumer GPU. So the edit passes through a rounding step before anyone uses it, and whether "
          "it survives that step is rarely checked.",
          [("Why study the removal of safety at all?",
            "The same question applies to edits that strengthen refusal. We only reproduced an edit that was "
            "already public, and we release no harmful outputs.")])
    return s


@add
def s03_question(F):
    s = content_slide("Research question", "One question runs through the whole report", 3)
    box(s, 0.6, 1.95, 12.1, 2.05, fill=TINT)
    text(s, 1.0, 2.1, 11.3, 1.8,
         "When an abliteration edit is applied at full precision and the model is then shrunk to 4-bit NF4, "
         "does the edit’s effect survive? And can any loss be told apart from ordinary quantization damage?",
         size=24, font=HEAD, italic=True, color=ACCENT_DK, anchor=MSO_ANCHOR.MIDDLE)
    cards = [
        ("Part 1: survival", "After 4-bit conversion, does the edited model still behave like the edited model?"),
        ("Part 2: attribution", "If something changes, was the edit lost, or does 4-bit rounding hurt every model a little?"),
    ]
    for i, (h, b) in enumerate(cards):
        x = 0.6 + i * 6.2
        box(s, x, 4.4, 5.9, 2.15, fill=LIGHT)
        text(s, x + 0.3, 4.6, 5.3, 0.5, h, size=20, bold=True, color=ACCENT_DK)
        text(s, x + 0.3, 5.15, 5.3, 1.3, b, size=18)
    notes(s, "Omm",
          "This is our one research question, and it has two parts. First, after the edited model is converted to "
          "4-bit NF4, does it still behave like the edited model? Second, if something changes, is that because the "
          "edit was lost, or simply because quantization damages every model a little? To separate those two, we "
          "also need an unedited model that is quantized in the same way. Everything in the report serves this "
          "question. The reproduction and the measurement work are not side projects: we needed them before we "
          "could answer it.",
          [("Why 4-bit, and why NF4?",
            "4-bit is how these models are commonly run. NF4 is a standard 4-bit format introduced with QLoRA, "
            "and we used its bitsandbytes implementation.")])
    return s


@add
def s04_direction(F):
    s = content_slide("Background: abliteration", "A refusal direction is the gap between two averages", 4)
    # left: point clouds
    box(s, 0.6, 1.9, 5.9, 3.85, fill=LIGHT)
    rnd = random.Random(7)
    hx, hy = 1.95, 4.55   # harmless centre
    fx, fy = 4.85, 2.85   # harmful centre
    for cx, cy, col in ((hx, hy, GREY), (fx, fy, ACCENT)):
        for _ in range(16):
            a = rnd.uniform(0, 2 * math.pi)
            r = abs(rnd.gauss(0, 0.42))
            px, py = cx + r * math.cos(a), cy + r * 0.75 * math.sin(a)
            box(s, px - 0.06, py - 0.06, 0.12, 0.12, fill=col, shape=MSO_SHAPE.OVAL)
    for cx, cy in ((hx, hy), (fx, fy)):
        box(s, cx - 0.13, cy - 0.13, 0.26, 0.26, fill=WHITE, line=INK, line_w=2.5, shape=MSO_SHAPE.OVAL)
    line(s, hx + 0.15, hy - 0.1, fx - 0.17, fy + 0.11, color=INK, width=3.0, head=True)
    text(s, 0.85, 5.1, 2.6, 0.55, "harmless prompts", size=16, color=MUTED, bold=True)
    text(s, 3.85, 2.0, 2.5, 0.4, "harmful prompts", size=16, color=ACCENT_DK, bold=True, align=PP_ALIGN.RIGHT)
    text(s, 3.45, 3.95, 2.3, 0.7, "refusal direction", size=17, bold=True, italic=True)
    text(s, 0.85, 2.0, 2.6, 0.4, "rings = averages", size=15, color=MUTED, italic=True)
    # equation 1 of 2
    box(s, 0.6, 5.95, 5.9, 0.75, fill=TINT)
    text(s, 0.6, 5.95, 5.9, 0.75,
         [{"t": [("refusal direction", {"bold": True}), (" = average(harmful) − average(harmless)", {})]}],
         size=18, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, color=ACCENT_DK)
    # right: steps
    steps = [
        "Run 400 harmful and 400 harmless prompts and record the model’s internal numbers (activations) at each layer.",
        "Average each group. The arrow from one average to the other is the refusal direction.",
        "Edit each layer’s output weights so the layer can no longer write along that direction.",
    ]
    for i, t in enumerate(steps):
        y = 2.0 + i * 1.6
        circle_num(s, 7.0, y, 0.55, str(i + 1))
        text(s, 7.8, y - 0.05, 4.9, 1.45, t, size=19)
    notes(s, "Omm",
          "Inside the model, each word is represented by a long list of numbers, 4096 of them in this model. We can "
          "treat that list as a point in space. We run 400 harmful and 400 harmless prompts, record these points at "
          "each layer, and average each group. The arrow from the harmless average to the harmful average is the "
          "refusal direction. Abliteration then edits the output weights of each layer so that the layer can no "
          "longer write anything along that arrow. If the model cannot push its state in that direction, it tends "
          "not to refuse.",
          [("Is refusal really a single direction?",
            "Not entirely. Later work found several related directions, which is why we check the model's actual "
            "answers instead of trusting the geometry."),
           ("Which layer's direction was used?",
            "The published settings pick a point between the outputs of blocks 18 and 19, mixed with weight 0.49.")])
    return s


NF4_LEVELS = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]


@add
def s05_nf4(F):
    s = content_slide("Background: quantization", "NF4 rounds every weight to one of 16 levels", 5)
    x0, x1, ly = 1.3, 12.0, 3.35

    def X(v):
        return x0 + (v + 1) / 2 * (x1 - x0)

    line(s, x0, ly, x1, ly, color=INK, width=2.0)
    for v in NF4_LEVELS:
        box(s, X(v) - 0.07, ly - 0.07, 0.14, 0.14, fill=ACCENT_DK, shape=MSO_SHAPE.OVAL)
    rnd = random.Random(3)
    ws = sorted(max(-0.93, min(0.93, rnd.gauss(0, 0.38))) for _ in range(13))
    for w in ws:
        q = min(NF4_LEVELS, key=lambda v: abs(v - w))
        line(s, X(w), 2.45, X(q), ly - 0.1, color=GREY, width=1.25, head=True, head_size="sm")
        box(s, X(w) - 0.07, 2.38, 0.14, 0.14, fill=GREY, shape=MSO_SHAPE.OVAL)
    text(s, 0.6, 1.85, 6.0, 0.45, "original 16-bit weights (any value)", size=16, color=MUTED, italic=True)
    text(s, X(-1) - 0.5, ly + 0.12, 1.0, 0.4, "−1", size=16, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, X(0) - 0.5, ly + 0.12, 1.0, 0.4, "0", size=16, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, X(1) - 0.5, ly + 0.12, 1.0, 0.4, "+1", size=16, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 3.2, ly + 0.5, 6.9, 0.45, "16 NF4 levels, packed tighter near zero where most weights sit",
         size=16, color=ACCENT_DK, italic=True, align=PP_ALIGN.CENTER)
    cards = [
        ("Group", "Weights are handled in blocks of 64."),
        ("Scale", "Each block is divided by its largest absolute value."),
        ("Round", "Each weight snaps to the nearest of 16 levels: 4 bits."),
    ]
    for i, (h, b) in enumerate(cards):
        x = 0.6 + i * 4.1
        box(s, x, 4.55, 3.8, 1.45, fill=TINT)
        text(s, x + 0.25, 4.65, 3.3, 0.45, h, size=19, bold=True, color=ACCENT_DK)
        text(s, x + 0.25, 5.1, 3.3, 0.9, b, size=17)
    text(s, 0.6, 6.2, 12.1, 0.6,
         "At run time the weights are expanded back to 16-bit, but the rounding error stays. Could it blur the edit away?",
         size=17, color=MUTED)
    notes(s, "Omm",
          "Quantization means storing each weight with fewer bits. NF4 groups the weights into blocks of 64. Each "
          "block is scaled by its largest absolute value, and then each weight is rounded to the nearest of 16 "
          "fixed levels. The levels sit closer together near zero, because most weights are small. When the model "
          "runs, the weights are expanded back to 16-bit for computation, but the rounding error stays. The worry is "
          "simple: the abliteration edit is a small change to the weights, so rounding might blur it away.",
          [("Why are the levels uneven?",
            "Weights are roughly normally distributed, so NF4 places its 16 levels at quantiles of a normal "
            "distribution, giving more levels where weights are dense."),
           ("Which settings did you use?",
            "bitsandbytes NF4 with blocks of 64, no double quantization and FP16 computation.")])
    return s


@add
def s06_steps(F):
    s = content_slide("Approach", "We answer it in three steps; the first two are preconditions", 6)
    steps = [
        ("Get a faithful edit", "Can we rebuild the published edit exactly enough to study it?", "precondition"),
        ("Get a trustworthy measure", "Does “no refusal phrase” really mean the model answered?", "precondition"),
        ("Test survival under NF4", "Does the edit’s effect stay after 4-bit conversion, and is any change due to the edit?",
         "the question"),
    ]
    w, gap, y, h = 3.6, 0.65, 2.05, 3.6
    for i, (head, q, tag) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        last = i == 2
        box(s, x, y, w, h, fill=TINT2 if last else TINT, line=ACCENT if last else None, line_w=2.0)
        circle_num(s, x + 0.3, y + 0.3, 0.65, str(i + 1), size=24)
        text(s, x + 1.1, y + 0.38, w - 1.3, 0.5, f"Step {i + 1}", size=16, bold=True, color=ACCENT)
        text(s, x + 0.3, y + 1.15, w - 0.55, 0.85, head, size=21, bold=True, font=HEAD,
             color=ACCENT_DK if last else INK)
        text(s, x + 0.3, y + 2.0, w - 0.55, 1.5, q, size=17)
        if i < 2:
            chevron(s, x + w + 0.2, y + h / 2 - 0.22, 0.26, 0.44)
    text(s, 0.6, 5.95, 12.1, 0.7,
         "Without step 1, a change tells us nothing about the real method. Without step 2, we could not see the change.",
         size=18, color=MUTED)
    notes(s, "Omm",
          "We split the question into three steps. Step one: we need a faithful copy of the published edit, "
          "otherwise any change after quantization tells us nothing about the real method. Step two: we need a "
          "measure we can trust. If we only count refusal phrases, a model that avoids refusal words but still does "
          "not answer would look fully edited. Step three is the actual test. Steps one and two are preconditions, "
          "and as you will see, they turned out to be the hard part. I now hand over to Shreyash.",
          [("Why not just use the publisher's checkpoint?",
            "For the five-condition design we need the edit as separate factors, so we can also apply it on top of "
            "a quantized base. That meant rebuilding it.")])
    return s


@add
def s07_setup(F):
    s = content_slide("Setup", "One hybrid model, one published edit, two Kaggle GPUs", 7)
    picture(s, F["fig3"], 0.5, 1.85, w=7.6,
            alt="Layer map of Qwen3.5-9B with the ablation strength of each edited attention and MLP matrix.")
    text(s, 8.45, 1.95, 4.3, 4.9, [
        {"t": [("Qwen3.5-9B: ", {"bold": True}), ("32 layers, three linear-attention layers to every one full-attention layer", {})], "bullet": True},
        {"t": [("Published Heretic edit ", {"bold": True}), ("rebuilt from its released settings, with no new search", {})], "bullet": True},
        {"t": [("42 matrices edited: ", {"bold": True}), ("19 attention outputs and 23 MLP outputs", {})], "bullet": True},
        {"t": [("Compute: ", {"bold": True}), ("two NVIDIA T4 GPUs on Kaggle, 16-bit", {})], "bullet": True},
    ], size=18, space_after=14)
    notes(s, "Shreyash",
          "Our model is Qwen3.5-9B. It has 32 layers and it is a hybrid: three linear-attention layers, called Gated "
          "DeltaNet, then one full-attention layer, repeated. The strip at the top of the figure shows this pattern. "
          "We studied a published Heretic edit of this model. The bars show how strongly each layer was edited: blue "
          "for the 19 attention output matrices and orange for the 23 MLP output matrices, 42 in total. Strengths "
          "above 1 go further than removal and reverse part of the component. Everything ran in 16-bit on two Kaggle T4 GPUs.",
          [("Why does the hybrid design matter?",
            "Both layer types end in an output projection, so the same editing rule applies, but we needed a Heretic "
            "revision that supports this architecture."),
           ("How much compute did it take?",
            "About 2.11 GPU-hours for the reconstruction, 1.04 for the NF4 pilot and 0.87 for the semantic audit.")])
    return s


@add
def s08_projection(F):
    s = content_slide("Step 1 result", "The first rebuild failed; one missing step fixed it", 8)
    picture(s, F["fig2"], 1.35, 1.65, w=10.6,
            alt="Left: raw and projected refusal directions. Right: marker matches fall from 72 to 8 and benign KL from 0.434 to 0.078.")
    # equation 2 of 2
    box(s, 0.6, 5.72, 12.1, 0.6, fill=TINT)
    text(s, 0.6, 5.72, 12.1, 0.6,
         [{"t": [("projected direction", {"bold": True}),
                 (" = raw direction − its part along the harmless average", {})]}],
         size=18, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, color=ACCENT_DK)
    text(s, 0.6, 6.47, 12.1, 0.45,
         "Why: every prompt shares the harmless part, so removing it disturbed all inputs and wasted part of the edit.",
         size=17, color=MUTED)
    notes(s, "Shreyash",
          "Our first attempt to rebuild the edit failed. With the published settings, 72 of 100 harmful test prompts "
          "still produced refusal phrases, and the change on harmless prompts, measured by KL divergence, was 0.434. "
          "The missing piece was one step known as projected abliteration: before editing, remove from the raw "
          "direction its part along the harmless average. With only that change, refusal phrases fell to 8 of 100 "
          "and KL to 0.078, the same as the publisher's checkpoint in our setup. The raw gap contains a part every "
          "prompt shares, so removing it disturbs all inputs and spends part of the edit on the wrong thing.",
          [("What is KL divergence here?",
            "How much the model's probabilities for the first output token on harmless prompts change compared with "
            "the original. Lower means less collateral change."),
           ("Did you test the fix separately?",
            "Yes, on 32 development prompts: marker matches went from 20 of 32 to 0 of 32, and KL from 0.438 to 0.066.")])
    return s


@add
def s09_checks(F):
    s = content_slide("Step 1 checks", "The rebuilt edit is verified, but phrase counts are not answers", 9)
    stats = [
        ("42 of 42", "edited matrices equal an explicit computation of original + edit"),
        ("100 of 100", "identical outputs after saving and reloading the model"),
        ("> 0.9999", "cosine match with the publisher’s change (MLP and full-attention matrices)"),
    ]
    for i, (big, lab) in enumerate(stats):
        x = 0.6 + i * 4.1
        box(s, x, 1.95, 3.8, 2.35, fill=TINT)
        text(s, x + 0.25, 2.05, 3.3, 0.95, big, size=44, bold=True, color=ACCENT, font=HEAD)
        text(s, x + 0.25, 3.0, 3.3, 1.2, lab, size=17)
    box(s, 0.6, 4.65, 12.1, 1.95, fill=WHITE, line=ACCENT, line_w=2.0)
    circle_num(s, 0.95, 5.0, 0.6, "!", size=24)
    text(s, 1.85, 4.85, 10.6, 0.55, "Caution: 8 refusal phrases does not mean 92 answers",
         size=21, bold=True, color=ACCENT_DK)
    text(s, 1.85, 5.45, 10.6, 1.1,
         "With a 100-token limit, 99 of 100 replies were cut off. Whether the model actually answered needs a "
         "different measure: that is Step 2.", size=18)
    notes(s, "Shreyash",
          "We also checked the rebuilt edit directly in the weights. All 42 edited matrices equal an explicit "
          "computation of the original weights plus the edit. Saving and reloading the model gave 100 of 100 "
          "identical outputs. Spot checks against the publisher's weights showed the change pointing the same way, "
          "with cosine similarity above 0.9999 for the MLP and full-attention matrices. One caution: our test used a "
          "100-token limit and 99 of 100 replies were cut off. So 8 refusal phrases does not mean 92 real answers. "
          "That is what step two is about.",
          [("Is your reconstruction identical to the publisher's model?",
            "Close but not byte for byte, because the publisher's exact unrounded settings and software environment "
            "were not released.")])
    return s


@add
def s10_markers(F):
    s = content_slide("Step 2: the measure", "Counting refusal phrases misleads in both directions", 10)
    gx, gy = 3.35, 1.95
    cw, ch, gap = 4.6, 1.7, 0.2
    text(s, gx, gy, cw, 0.4, "Gave the requested content", size=17, bold=True, color=ACCENT_DK, align=PP_ALIGN.CENTER)
    text(s, gx + cw + gap, gy, cw, 0.4, "Did not give it", size=17, bold=True, color=ACCENT_DK, align=PP_ALIGN.CENTER)
    rows = ["Uses refusal words", "No refusal words"]
    cells = [
        [("Refuses, then answers anyway", "Counter says “refusal”: wrong", "seen in the original model", True),
         ("Plain refusal", "Counter is right", "", False)],
        [("Plain answer", "Counter is right", "", False),
         ("Polite redirect", "Counter says “answered”: wrong", "seen in an edited model", True)],
    ]
    for r in range(2):
        y = gy + 0.5 + r * (ch + gap)
        text(s, 0.6, y, 2.6, ch, rows[r], size=18, bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)
        for c in range(2):
            x = gx + c * (cw + gap)
            head, verdict, where, hl = cells[r][c]
            box(s, x, y, cw, ch, fill=TINT2 if hl else LIGHT, line=ACCENT if hl else None, line_w=2.0)
            paras = [{"t": head, "size": 20, "bold": True, "color": ACCENT_DK if hl else INK},
                     {"t": verdict, "size": 17, "color": INK if hl else MUTED}]
            if where:
                paras.append({"t": where, "size": 16, "italic": True, "color": MUTED})
            text(s, x + 0.3, y + 0.2, cw - 0.6, ch - 0.3, paras, space_after=4)
    text(s, 0.6, 6.35, 12.1, 0.5,
         "A phrase counter reads the wording. We need to judge what the reply actually gives.", size=17, color=MUTED)
    notes(s, "Shreyash",
          "Most abliteration tools count refusal phrases such as \"sorry\" or \"I can't\". That only reads the "
          "wording. This grid shows why that is not enough. The two plain cases are fine. But we saw two misleading "
          "cases. The original model once began with a refusal and then supplied the requested material anyway, and "
          "a phrase counter calls that a refusal. One edited reply had no refusal words at all but steered the user "
          "toward protecting their own account, and a counter calls that an answer. So we switched to semantic "
          "review: does the reply actually give what was asked?",
          [("Who did the semantic labeling?",
            "A single LLM-based reviewer working on shuffled, anonymized responses, with labels fixed before the "
            "conditions were revealed. Human labels are future work."),
           ("Does a disclaimer cancel an answer?",
            "No. In our rubric a disclaimer does not cancel fulfillment, and an attempt is not checked for correctness.")])
    return s


@add
def s11_semantic(F):
    s = content_slide("Step 2 result", "Judged by content, the edit clearly works at full precision", 11)
    picture(s, F["fig4"], 0.45, 1.8, w=8.25,
            alt="Semantic fulfillment for original, reconstruction and publisher models on matched and fresh safety prompts, next to two refusal detectors.")
    x = 9.05
    box(s, x, 1.95, 3.7, 1.75, fill=TINT)
    text(s, x + 0.25, 2.0, 3.3, 0.8, "6 of 8", size=40, bold=True, color=ACCENT, font=HEAD)
    text(s, x + 0.25, 2.85, 3.3, 0.8, "fresh safety prompts clearly answered by each edited model", size=16)
    box(s, x, 3.9, 3.7, 1.45, fill=LIGHT)
    text(s, x + 0.25, 3.95, 3.3, 0.8, "1 of 8", size=40, bold=True, color=MUTED, font=HEAD)
    text(s, x + 0.25, 4.8, 3.3, 0.5, "by the original model", size=16)
    text(s, x, 5.55, 3.75, 1.3,
         "Our rebuild and the publisher’s model got the same label on all 14 prompts.", size=16, color=MUTED)
    notes(s, "Shreyash",
          "Here are the semantic labels. On eight fresh safety prompts, the original model clearly answered one and "
          "used refusal language on all eight. Both edited models, our rebuild and the publisher's, clearly answered "
          "six and partly answered two. They received the same label on all 14 prompts. The small columns on the "
          "right show the two phrase-based detectors on the same replies, and they do not line up with the bars. "
          "From here on, our outcome is semantic fulfillment, not phrase counts.",
          [("Two fresh prompts were about security testing. Does that inflate the result?",
            "Excluding them leaves 4 clear and 2 partial answers out of 6 for each edited model, and none for the "
            "original."),
           ("Did the edit damage general ability?",
            "No model refused the safe XSTest prompts and all three solved the five arithmetic problems, but that "
            "check is too small to prove ability is preserved.")])
    return s


@add
def s12_design(F):
    s = content_slide("Step 3: the design", "Five versions separate edit loss from rounding damage", 12)
    picture(s, F["fig1"], 0.45, 1.95, w=6.75,
            alt="Five-condition design: original and edited models at full precision and after NF4, plus NF4 base with unrounded edit.")
    rows = [
        ("W", "original, 16-bit", False),
        ("W + Δ", "edited, 16-bit", False),
        ("Q(W)", "original, 4-bit: the control", True),
        ("Q(W + Δ)", "edited, then 4-bit: deployment", False),
        ("Q(W) + Δ", "4-bit base, edit kept unrounded", False),
    ]
    x = 7.55
    for i, (lab, desc, hl) in enumerate(rows):
        y = 1.95 + i * 0.62
        box(s, x, y, 5.2, 0.52, fill=TINT2 if hl else LIGHT, line=ACCENT if hl else None, line_w=1.5)
        text(s, x + 0.2, y, 1.6, 0.52, lab, size=17, bold=True, color=ACCENT_DK, anchor=MSO_ANCHOR.MIDDLE,
             font=HEAD)
        text(s, x + 1.8, y, 3.35, 0.52, desc, size=16, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, 5.15, 5.2, 1.7, [
        {"t": "Why the control matters", "size": 18, "bold": True, "color": ACCENT_DK, "space_after": 4},
        {"t": "If Q(W) started answering too, 4-bit rounding, not the edit, could explain the result.", "size": 17},
    ])
    notes(s, "Shreyash",
          "Now the design for step three. We compare five versions of the model. The top row of the picture is full "
          "precision: original and edited. The bottom row is after 4-bit conversion. The key control is the original "
          "model, quantized. If quantization alone made the model start answering, we would wrongly credit the edit. "
          "The fifth version keeps the edit in 32-bit floating point on top of the quantized base, so the edit itself "
          "is never rounded. The question is whether the gap between the two bottom boxes matches the gap between "
          "the two top boxes. Aman will present the result.",
          [("Why the fifth condition?",
            "It separates rounding of the edit itself from changes in the base model, because the edit factors stay "
            "in FP32."),
           ("Did you re-extract the direction from the 4-bit model?",
            "No. That is a different experiment and is listed as future work.")])
    return s


@add
def s13_result(F):
    s = content_slide("Step 3 result", "The edit’s clear answers survive NF4; NF4 alone creates none", 13)
    picture(s, F["fig5"], 0.45, 1.95, w=8.6,
            alt="NF4 pilot: clear, partial, unknown and no fulfillment for the five conditions on six safety prompts.")
    x = 9.4
    items = [
        ("2 → 2", "clear answers before and after NF4", ACCENT, TINT),
        ("0 → 0", "for the original model: NF4 alone creates none", ACCENT, TINT),
        ("Open", "partial weakening not ruled out; most edited replies were cut off", MUTED, LIGHT),
    ]
    for i, (big, lab, col, fill) in enumerate(items):
        y = 1.95 + i * 1.62
        box(s, x, y, 3.35, 1.5, fill=fill)
        text(s, x + 0.25, y + 0.05, 3.0, 0.6, big, size=28, bold=True, color=col, font=HEAD)
        text(s, x + 0.25, y + 0.62, 2.95, 0.85, lab, size=16)
    text(s, 0.6, 6.35, 8.4, 0.5, "Six safety prompts per condition: a pilot, not a rate.", size=17, color=MUTED)
    notes(s, "Aman",
          "This is the main result, on six safety prompts per condition. The original model and its 4-bit version "
          "look the same, with no clear answers, so 4-bit conversion alone does not create the effect. The edited "
          "model clearly answered two prompts at full precision, and those same two were still clearly answered after "
          "quantization, and also when the edit was kept unrounded. No quantized edited version had fewer clear or "
          "partial answers than the full-precision edit. But most edited replies were cut off and some labels are "
          "unknown, so we can rule out complete erasure on these prompts, not partial weakening.",
          [("How much could the effect have changed?",
            "If every partial or unknown label could go either way, the change in the edit's effect lies between "
            "−0.50 and +0.67. These are bounds over possible labelings, not confidence intervals."),
           ("Did quantization hurt the benign prompts?",
            "No condition refused the three safe prompts. Four conditions solved all three arithmetic problems; the "
            "unrounded-edit condition solved two and hit its limit on the third.")])
    return s


@add
def s14_hypothesis(F):
    s = content_slide("Discussion", "Why it might survive: a coherent push beats scattered noise", 14)
    pill = box(s, 9.6, 0.36, 3.1, 0.42, fill=WHITE, line=ACCENT, line_w=1.5, radius=0.5)
    text(s, 9.6, 0.36, 3.1, 0.42, "Hypothesis, not measured", size=16, bold=True, color=ACCENT,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    panels = [("The edit", "one large push along one direction"),
              ("Rounding error", "many small pushes in every direction")]
    for i, (h, sub) in enumerate(panels):
        x = 0.6 + i * 6.2
        box(s, x, 1.9, 5.9, 3.05, fill=LIGHT)
        text(s, x + 0.3, 2.0, 5.3, 0.45, h, size=20, bold=True, color=ACCENT_DK)
        text(s, x + 0.3, 2.45, 5.3, 0.4, sub, size=16, color=MUTED, italic=True)
        cy = 3.85
        line(s, x + 0.4, cy, x + 5.5, cy, color=ACCENT, width=1.5, dash=7)  # refusal direction, dashed
        text(s, x + 3.6, cy + 0.6, 2.1, 0.35, "refusal direction", size=16, color=ACCENT, italic=True,
             align=PP_ALIGN.RIGHT)
        if i == 0:
            line(s, x + 0.9, cy, x + 4.9, cy, color=ACCENT_DK, width=7, head=True, head_size="med")
        else:
            rnd = random.Random(11)
            cx = x + 2.95
            for k in range(22):
                a = rnd.uniform(0, 2 * math.pi)
                L = rnd.uniform(0.25, 0.6)
                ox, oy = cx + rnd.uniform(-1.6, 1.6), cy + rnd.uniform(-0.4, 0.4)
                line(s, ox, oy, ox + L * math.cos(a), oy + L * 0.8 * math.sin(a), color=GREY, width=1.5,
                     head=True, head_size="sm")
    text(s, 0.6, 5.15, 12.1, 1.8, [
        {"t": "The edit has strength 0.84 to 2.0, all along one direction. Rounding errors are small and spread "
              "over 4096 directions, so only a tiny share lands on the refusal direction.", "bullet": True},
        {"t": "Prediction: round-to-nearest 4-bit keeps the edit, unless the strength is small or the format is much coarser.",
         "bullet": True},
    ], size=18, space_after=8)
    notes(s, "Aman",
          "Why might the edit survive? This is our hypothesis, not a measured result. The edit changes each edited "
          "matrix by a large amount, with strength between 0.84 and 2.0, and always along one direction: a single "
          "coherent push. Rounding errors are small and point in all directions across the 4096-dimensional space, "
          "so only a tiny share of them falls along the refusal direction. Coherent signal beats scattered noise. "
          "This predicts that round-to-nearest 4-bit should keep the edit, unless the edit is weak or the format is "
          "much coarser. It also suggests a cheap test that needs only the weights.",
          [("How would you test it?",
            "For each edited matrix, measure what fraction of the intended edit is still present in the difference "
            "between the quantized edited and quantized original weights, for several bit widths. No text "
            "generation is needed."),
           ("Is the hypothesis proven?",
            "No. It is an argument from the structure of the edit. The weight-only test on the future-work slide would check it.")])
    return s


@add
def s15_limits(F):
    s = content_slide("Limitations", "What this pilot cannot tell us", 15)
    cards = [
        ("Small samples", "14 safety prompts at 16-bit and 6 in the NF4 pilot. No rates or confidence intervals."),
        ("One AI reviewer", "All semantic labels came from one LLM reviewer with imperfect masking. No human labels yet."),
        ("Truncated answers", "Most edited replies hit the token limit, so labels cover only what was visible."),
        ("One model, one edit", "Only Qwen3.5-9B and one published Heretic edit were studied."),
        ("One quantizer", "Only bitsandbytes NF4, in one configuration. Other formats may differ."),
        ("Not checked", "Correctness of harmful content, on purpose. The 100-prompt test was also used while debugging."),
    ]
    w, h = 3.85, 2.2
    for i, (head, body) in enumerate(cards):
        r, c = divmod(i, 3)
        x, y = 0.6 + c * (w + 0.275), 1.95 + r * (h + 0.3)
        box(s, x, y, w, h, fill=TINT if i < 3 else LIGHT)
        circle_num(s, x + 0.25, y + 0.25, 0.45, str(i + 1), size=16, fill=ACCENT if i < 3 else MUTED)
        text(s, x + 0.85, y + 0.25, w - 1.0, 0.5, head, size=19, bold=True, color=ACCENT_DK if i < 3 else INK)
        text(s, x + 0.25, y + 0.85, w - 0.45, 1.3, body, size=16)
    notes(s, "Aman",
          "These are the limits on what we can claim. The samples are small: 14 safety prompts at full precision "
          "and 6 in the NF4 pilot, so we report no rates or confidence intervals. All semantic labels came from one "
          "AI reviewer, and the masking was imperfect. Most edited replies hit the token limit, so we labeled only "
          "what was visible. We studied one model, one edit and one quantizer configuration. We deliberately did not "
          "check whether harmful answers were correct, and the 100-prompt reproduction test was also used while "
          "debugging.",
          [("Given all this, what can you actually claim?",
            "That complete erasure of the edit did not happen on these prompts with this setup. Not that the edit is "
            "fully preserved in general.")])
    return s


@add
def s16_ethics(F):
    s = content_slide("Ethics and disclosure", "We reproduced only public work and show no harmful text", 16)
    cols = [
        ("Ethics", [
            "Only an edit and a checkpoint that were already public were reproduced.",
            "No harmful model output is shown; replies are described by category only.",
            "Raw outputs and the rebuilt model are kept private and not released.",
            "Correctness of harmful answers was deliberately not tested.",
        ]),
        ("AI assistance", [
            "AI coding agents helped write the code and run the experiments.",
            "Semantic labels came from a single LLM-based reviewer.",
            "All responses and settings were saved for later human review.",
        ]),
    ]
    for i, (head, items) in enumerate(cols):
        x = 0.6 + i * 6.2
        box(s, x, 1.95, 5.9, 4.55, fill=TINT if i == 0 else LIGHT)
        circle_num(s, x + 0.3, 2.2, 0.55, "E" if i == 0 else "AI", size=17, fill=ACCENT if i == 0 else MUTED)
        text(s, x + 1.05, 2.22, 4.6, 0.5, head, size=22, bold=True, font=HEAD, color=ACCENT_DK if i == 0 else INK)
        text(s, x + 0.35, 3.0, 5.25, 3.4, [{"t": t, "bullet": True} for t in items], size=17, space_after=10)
    notes(s, "Aman",
          "Because abliteration removes a safety behavior, this work is dual-use, so we set clear limits. We "
          "reproduced only an edit and a checkpoint that were already public. No harmful model output appears in "
          "the report or in these slides; we describe replies only by category. The raw outputs and the rebuilt model "
          "are in private, access-controlled storage and are not released. We did not test whether harmful answers "
          "were correct. We also disclose that AI coding agents helped with the implementation and the experiment "
          "runs, and that the semantic labels came from a single LLM-based reviewer.",
          [("Why is this research useful rather than harmful?",
            "It tells people whether a deployed 4-bit model still carries an edit, and the same question matters for "
            "edits that strengthen refusal."),
           ("Did a human check any labels?",
            "Not yet. All 132 responses are saved so that people can label them, which is our second next step.")])
    return s


@add
def s17_conclusion(F):
    s = content_slide("Conclusion", "The edit survived NF4 here; the hard part was measuring it", 17)
    box(s, 0.6, 1.95, 12.1, 1.05, fill=TINT2)
    text(s, 0.95, 1.95, 11.4, 1.05,
         "On Qwen3.5-9B with NF4, the edit’s observed effect survived quantization. Whether it is partly weakened "
         "remains open at this sample size.", size=19, font=HEAD, italic=True, color=ACCENT_DK, anchor=MSO_ANCHOR.MIDDLE)
    cards = [
        ("Rebuild every step", "One missing projection step was the difference between 72 and 8 refusal phrases out of 100."),
        ("Measure answers, not phrases", "Phrase counters err both ways. Judge what each reply actually gives."),
        ("The edit survived NF4", "Both clear answers stayed after 4-bit conversion, and NF4 alone created none."),
    ]
    w = 3.85
    for i, (h, b) in enumerate(cards):
        x = 0.6 + i * (w + 0.275)
        box(s, x, 3.3, w, 3.25, fill=LIGHT)
        circle_num(s, x + 0.3, 3.55, 0.6, str(i + 1), size=20)
        text(s, x + 0.3, 4.3, w - 0.55, 0.85, h, size=20, bold=True, font=HEAD)
        text(s, x + 0.3, 5.15, w - 0.55, 1.35, b, size=17)
    notes(s, "Aman",
          "To conclude, three takeaways. First, rebuilding a published edit needs every step: one missing projection "
          "step was the difference between 72 and 8 refusal phrases out of 100. Second, measure answers, not phrases. "
          "Phrase counters make errors in both directions, so we judged what each reply actually gave. Third, in our "
          "pilot the edit survived NF4: the clear answers seen at full precision stayed, and quantizing the original "
          "model did not create any. The lesson is that the hard part of the question was measurement, not compute.",
          [("Does this mean every 4-bit abliterated model is still abliterated?",
            "No. It is one model, one quantizer and a small pilot. Coarser formats may behave differently.")])
    return s


@add
def s18_future(F):
    s = content_slide("Future work", "Three next steps, cheapest first", 18)
    steps = [
        ("Weight-only retention test", "Measure how much of the edit remains in the 4-bit weights, for NF4, 8-bit and lower formats.",
         "runs on a CPU, no text generation"),
        ("Human labels", "People label the 132 saved responses, with two reviewers on a shared subset to measure agreement.",
         "checks the AI reviewer"),
        ("Larger study", "More prompts, longer answers, a calibrated evaluator such as StrongREJECT, a second quantizer and model.",
         "turns the pilot into rates"),
    ]
    w, gap, y, h = 3.6, 0.65, 1.95, 4.0
    for i, (head, body, tag) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        box(s, x, y, w, h, fill=TINT if i == 0 else LIGHT, line=ACCENT if i == 0 else None, line_w=2.0)
        circle_num(s, x + 0.3, y + 0.3, 0.6, str(i + 1), size=20, fill=ACCENT if i == 0 else MUTED)
        text(s, x + 0.3, y + 1.05, w - 0.55, 0.85, head, size=20, bold=True, font=HEAD,
             color=ACCENT_DK if i == 0 else INK)
        text(s, x + 0.3, y + 1.9, w - 0.55, 1.5, body, size=17)
        text(s, x + 0.3, y + h - 0.6, w - 0.55, 0.45, tag, size=16, italic=True, color=ACCENT if i == 0 else MUTED)
        if i < 2:
            chevron(s, x + w + 0.2, y + h / 2 - 0.22, 0.26, 0.44, color=GREY)
    text(s, 0.6, 6.2, 12.1, 0.6,
         "Also: re-extract the refusal direction from the 4-bit model, a condition this study did not run.",
         size=17, color=MUTED)
    notes(s, "Aman",
          "Three next steps, in order. First, a weight-only test: compare the quantized edited weights with the "
          "quantized original weights and measure how much of the intended edit is still there, for NF4, 8-bit and "
          "lower-precision formats. It runs on a CPU and directly tests our hypothesis. Second, human reviewers "
          "should label the 132 saved responses, with two reviewers on a shared subset to measure agreement. Third, "
          "a larger study: more prompts, longer answers, a calibrated evaluator such as StrongREJECT, a second "
          "quantizer and model family, and a condition that re-extracts the direction from the quantized model.",
          [("Why is the weight-only test first?",
            "It is the cheapest: it needs only the saved weights, no GPU and no text generation, and it directly "
            "tests the explanation on the previous slides.")])
    return s


@add
def s19_thanks(F):
    s = prs.slides.add_slide(BLANK)
    set_bg(s)
    text(s, 0.8, 0.9, 11.7, 1.0, "Thank you", size=44, bold=True, font=HEAD)
    text(s, 0.8, 1.85, 11.7, 0.6, "Questions are welcome", size=24, color=ACCENT_DK, font=HEAD)
    recap = [
        ("72 → 8", "refusal phrases out of 100 after the projection fix"),
        ("6 of 8 vs 1 of 8", "fresh prompts clearly answered: edited vs original"),
        ("2 → 2", "clear answers kept after NF4; none created by NF4 alone"),
    ]
    for i, (big, lab) in enumerate(recap):
        x = 0.8 + i * 3.95
        box(s, x, 2.85, 3.7, 1.9, fill=TINT)
        text(s, x + 0.25, 2.95, 3.3, 0.8, big, size=30, bold=True, color=ACCENT, font=HEAD)
        text(s, x + 0.25, 3.75, 3.25, 0.95, lab, size=16)
    names = [("Omm P Tripathy", "B123082"), ("Shreyash Jena", "B123124"), ("Aman Saini", "B423008")]
    for i, (n, r) in enumerate(names):
        text(s, 0.8 + i * 3.95, 5.15, 3.6, 0.8,
             [{"t": n, "size": 19, "bold": True, "space_after": 2}, {"t": r, "size": 16, "color": MUTED}])
    text(s, 0.8, 6.1, 11.7, 0.8, [
        {"t": "Supervisor: Prof. Srichandan Sobhanayak", "size": 16, "space_after": 2},
        {"t": "Department of Computer Science, IIIT Bhubaneswar  |  November 2026", "size": 16, "color": MUTED},
    ])
    notes(s, "Aman (all three answer questions)",
          "Thank you for listening. To sum up in three numbers: the projection fix took refusal phrases from 72 to 8 "
          "out of 100. On fresh prompts the edited models clearly answered 6 of 8, against 1 of 8 for the original. "
          "And the 2 clear answers seen at full precision were kept after NF4, while NF4 alone created none. We are "
          "happy to take questions. Omm will take background and motivation, Shreyash the reproduction and "
          "measurement, and Aman the NF4 results and limitations.",
          [("Why Qwen3.5-9B?",
            "It is a recent hybrid model with a published Heretic edit and released edit settings, and it runs on the "
            "two T4 GPUs we had."),
           ("What would change your conclusion?",
            "Human labels that disagree with the AI reviewer, or the weight-only test showing that much of the edit "
            "is lost in rounding.")])
    return s


@add
def s20_backup(F):
    s = content_slide("Backup", "Reproduction numbers on the 100-prompt test", "Backup")
    hdr = ["Condition", "Refusal-phrase matches", "Reached 100-token limit", "Benign KL"]
    rows = [
        ["Original W", "100", "59", "0"],
        ["Raw-direction replay", "72", "91", "0.434"],
        ["Publisher P", "8", "100", "0.078"],
        ["Reconstruction D", "8", "99", "0.078"],
        ["D after save and reload", "8", "99", "not measured"],
        ["Publisher’s model card (reported)", "6", "n/a", "0.079"],
    ]
    widths = [4.6, 2.6, 2.7, 2.2]
    tshape = s.shapes.add_table(len(rows) + 1, 4, Inches(0.6), Inches(1.9), Inches(sum(widths)), Inches(0.5 * 7))
    tbl = tshape.table
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for i in range(len(rows) + 1):
        tbl.rows[i].height = Inches(0.5)
        for j in range(4):
            cell = tbl.cell(i, j)
            val = hdr[j] if i == 0 else rows[i - 1][j]
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            hl = i in (3, 4)
            _apply_run(r, 17, i == 0 or (hl and j == 0), WHITE if i == 0 else INK, False, BODY)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if i == 0 else (TINT2 if hl else (LIGHT if i % 2 == 0 else WHITE))
    text(s, 0.6, 5.65, 12.1, 1.2, [
        {"t": "Controlled check on 32 development prompts: 20 of 32 → 0 of 32 phrase matches; KL 0.438 → 0.066.",
         "bullet": True},
        {"t": "KL: change in the first-token prediction on 100 harmless prompts, compared with the original model.",
         "bullet": True},
    ], size=17, space_after=6, color=MUTED)
    notes(s, "Shreyash",
          "Backup slide for questions on the reproduction. The original model hits refusal phrases on all 100 test "
          "prompts. The raw-direction replay still hits 72. The publisher's checkpoint and our reconstruction both "
          "hit 8, with the same KL of 0.078, and after save and reload our reconstruction still hits 8. The "
          "publisher's model card reports 6 and 0.079 in their own setup. The limit column shows how many replies "
          "were cut off at 100 tokens, which is why these counts say little about real answers.",
          [("Why do the model card numbers differ slightly?",
            "They come from the publisher's own evaluation setup. We compare every model within one setup, where "
            "the publisher's checkpoint and our reconstruction match."),
           ("Were these test prompts held out?",
            "No. They were also used while debugging the reconstruction, which we list as a limitation.")])
    return s


# --------------------------------------------------------------------------- build and checks
BANNED = ["—", "–", "delve", "crucial", "robust", "landscape", "leverage", "seamless"]


def all_text(slide):
    out = []
    for sh in slide.shapes:
        if sh.has_text_frame:
            out.append(sh.text_frame.text)
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for c in row.cells:
                    out.append(c.text)
    out.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(out)


def main():
    F = prep_figures()
    for fn in SLIDES:
        fn(F)
    problems = []
    for i, slide in enumerate(prs.slides, 1):
        body = getattr(slide, "_note_body", "")
        n = len(body.split())
        if not 60 <= n <= 120:
            problems.append(f"slide {i}: notes body has {n} words")
        low = all_text(slide).lower()
        for b in BANNED:
            if b in low:
                problems.append(f"slide {i}: banned string {b!r}")
    prs.save(OUT)
    print(f"wrote {OUT} with {len(prs.slides)} slides")
    for p in problems:
        print("CHECK:", p)


if __name__ == "__main__":
    main()
