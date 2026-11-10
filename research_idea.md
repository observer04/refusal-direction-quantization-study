PROMPT: 
basic idea of research my team thought to pursue on what to do in the AI safety space for an undergrad research. now i'm thinking and you should refine and if required add/subtract to my idea. of doing this: first we'll take a small open source model as a standard then test a bunch of safety eval prompts on it to see how it behaves, and we'll somehow monitor the activations? then what we'll do is we'll abliterate or orthogonalize the model and produce both an offensive and a defensive model and then we'll put down the same tasks and see how it behaves. that's phase 2 and then for the next step ,we'll do some quantizations and again the same prompts/tasks to see how it behaves.. is it a good idea? is there area here where we could actually do some novel finding .

RESPONSE:
Yes — **this can become a strong undergraduate AI-safety research project**, but I would change the framing substantially.

The weak version is:

> baseline → abliterate → quantize → run safety prompts → compare scores.

That is mostly an empirical replication now. Refusal-direction ablation/weight orthogonalization is established by Arditi et al., and recent work has already shown that quantization can alter LLM safety. [arXiv](https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com)

The stronger version is:

> **What happens to the internal geometry of safety/refusal when an aligned LLM is compressed, and how does that affect its susceptibility to activation/weight interventions?**

That is much more interesting.

## The research question I would build around

A possible title:

**Quantization-Induced Drift in Refusal Representations of Safety-Aligned Language Models**

Or a little broader:

**Evaluating the Stability and Manipulability of Refusal Representations in Quantized Large Language Models**

Your core question becomes:

> **Does quantization preserve the internal representations responsible for refusal, or does it rotate, weaken, redistribute, or fragment them?**

And then:

> **Do these representational changes predict changes in actual safety behavior?**

That's a meaningful mechanistic-interpretability + AI-safety question.

Arditi et al. showed that much refusal behavior can be manipulated through a low-dimensional direction in the residual stream. [arXiv](https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com) More recent work complicates the simple “one universal refusal vector” picture: safety appears more multidimensional in some models, and different architectures can implement refusal at different depths. [arXiv](https://arxiv.org/abs/2502.09674?utm_source=chatgpt.com) That actually makes your project **more interesting**, because quantization may affect these geometries differently.

---

# What I would make the project look like

Instead of three independent phases, think of the project as one controlled experiment.

| Stage | What you study |
|---|---|
| Baseline | What safety/refusal looks like behaviorally and internally |
| Intervention | What happens when the refusal representation is suppressed or strengthened |
| Quantization | Whether compression changes that representation |
| Interaction | Whether quantization changes the effectiveness of the intervention |
| Generalization | Whether the same phenomenon appears in another model family |

### Stage 1 — Establish the baseline

Take a **small instruction-tuned open-weight model**.

I'd actually use **two** eventually, rather than one.

Something approximately in the 1–4B range is ideal for undergrad compute. More important than exact model choice is having **different model families**, because recent evidence suggests refusal can develop at different points in the network across architectures. [arXiv](https://arxiv.org/abs/2606.22686?utm_source=chatgpt.com)

For example, conceptually:

**Model A: Qwen-family small instruct model**

**Model B: Llama/Gemma-family small instruct model**

Don't start with six models. Get the whole pipeline working on one and then replicate on a second.

Run three categories of prompts:

**Unsafe prompts** → should refuse.

**Clearly benign prompts** → should answer.

**Benign-but-dangerous-looking prompts** → should answer, but aligned models sometimes over-refuse.

That third category is essential. Otherwise a model that says *“I refuse everything”* appears perfectly safe.

XSTest was specifically created to measure this exaggerated-safety/over-refusal problem. [arXiv](https://arxiv.org/abs/2308.01263?utm_source=chatgpt.com)

For unsafe behavior, use established evaluations rather than inventing 50 prompts yourselves. HarmBench, StrongREJECT and SORRY-Bench provide much stronger evaluation methodology. [arXiv](https://arxiv.org/abs/2402.04249?utm_source=chatgpt.com)

---

# Stage 2 — Look inside the model

This is where your idea becomes more than ordinary benchmarking.

For each layer, record residual-stream activations for your harmless and harmful prompt sets.

Very roughly, you have:

\[
h_l^{harmful}
\]

and

\[
h_l^{harmless}
\]

at layer \(l\).

A simple refusal-direction estimate is something like:

\[
r_l =
\operatorname{mean}(h_l^{harmful})
-
\operatorname{mean}(h_l^{harmless})
\]

which is closely related to the methodology used in the refusal-direction literature. [arXiv](https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com)

But **don't stop at extracting one vector**.

Measure how separable the representations are at every layer.

For example:

\[
\text{harmful}
\quad \text{vs.} \quad
\text{harmless}
\]

using:

- cosine projections onto the refusal direction;
- a simple linear probe;
- PCA for visualization;
- perhaps \(k\)-dimensional SVD/PCA subspaces rather than assuming safety is exactly rank-1.

That last point is increasingly important because newer research suggests refusal can involve several dimensions rather than exactly one axis. [arXiv](https://arxiv.org/abs/2502.09674?utm_source=chatgpt.com)

Your baseline plots might eventually look conceptually like:

```text
Layer
0    4    8    12    16    20    24
|----|----|-----|-----|-----|-----|

harmful/benign separation
       ▁▂▃▅██████▇▅

refusal-direction magnitude
       ▁▁▂▄▇████▆▃
```

Now you've characterized **where safety appears inside the network**.

---

# Stage 3 — Your “offensive” and “defensive” variants

I'd change the terminology in the paper.

Instead of:

> offensive model

call it:

**refusal-ablated condition**

or

**safety-suppressed intervention**

And instead of:

> defensive model

call it:

**refusal-enhanced condition**

or

**safety-steered condition**.

That sounds much more like legitimate mechanistic safety research.

Your three conditions become:

\[
M
\]

original model,

\[
A(M)
\]

refusal-ablated model/intervention,

and

\[
S(M)
\]

refusal-strengthened/steered model.

You're testing both directions of the same mechanism.

This is valuable because merely showing that removing a vector destroys refusal mostly reproduces Arditi et al. But showing how that mechanism **changes under compression** is more interesting. [arXiv](https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com)

---

# Stage 4 — Quantization is where your actual novelty can live

Here's the key change I'd make.

Don't merely ask:

> Does INT4 make safety worse?

That question has already received considerable attention. ICML 2025 work found safety degradation across mainstream quantization techniques and proposed quantization-aware safety restoration; subsequent work has similarly explored alignment-aware quantization. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/chen25ci.html?utm_source=chatgpt.com)

Instead ask:

## **What does quantization do to the refusal representation itself?**

Suppose the FP16 refusal vector is

\[
r_{16}
\]

and after 8-bit quantization it becomes

\[
r_8
\]

and after 4-bit:

\[
r_4.
\]

Calculate:

\[
\cos(r_{16},r_8)
\]

and

\[
\cos(r_{16},r_4).
\]

Now you're asking whether compression **rotates the safety representation**.

You might find:

```text
FP16 → INT8      cos = .96
FP16 → INT4      cos = .74
FP16 → INT2      cos = .41
```

That's hypothetical, but a result like that would be genuinely interesting.

And then ask:

> Does the amount of rotation predict the amount of behavioral safety degradation?

For example:

\[
\Delta \text{Safety}
\propto
1-\cos(r_{FP16},r_{quantized})
\]

If there were a strong relationship, that's potentially a very neat finding.

---

# There is an even better experiment hiding here

This is the one I'd be most excited about.

## **Does quantization commute with safety intervention?**

You have two transformations.

Abliteration:

\[
A(\cdot)
\]

and quantization:

\[
Q(\cdot).
\]

Normally people might assume:

\[
Q(A(M))
\approx
A(Q(M))
\]

But **is that actually true?**

Experimentally compare:

\[
\boxed{Q(A(M))}
\]

with

\[
\boxed{A(Q(M))}
\]

In words:

**Path 1**

```text
aligned model
     ↓
abliterate
     ↓
quantize
```

versus

**Path 2**

```text
aligned model
     ↓
quantize
     ↓
extract new refusal representation
     ↓
abliterate
```

These don't necessarily have to produce the same model.

And that gives you a beautiful research question:

> **Is refusal-direction manipulation invariant to quantization, or is the order of compression and safety intervention important?**

You could define something like an intervention-order gap:

\[
G =
\left|
Safety(Q(A(M)))
-
Safety(A(Q(M)))
\right|.
\]

If \(G\) is large, you've discovered that safety interventions and deployment compression are **non-commutative**.

That has a practical implication:

> A safety property verified before model compression may not remain equivalent after compression.

I searched specifically around refusal vectors/subspaces + quantization and did not find an obvious existing paper centered on this exact interaction. There is substantial adjacent work on quantization safety and substantial work on refusal geometry, but the **intersection appears much less crowded**. I wouldn't claim novelty formally until you do a proper literature review, but this is the direction I'd investigate first.

---

# Another potentially novel result: transferability

There's a second very nice question.

Extract the refusal vector from FP16:

\[
r_{FP16}
\]

Then apply **that exact vector** to the quantized representation.

Compare it against a refusal vector freshly estimated from the INT4 model:

\[
r_{INT4}.
\]

Then ask:

> **Does an FP16 refusal representation transfer to the quantized version of the same model?**

You could construct a matrix like:

| Vector extracted from ↓ / Model applied to → | FP16 | INT8 | INT4 |
|---|---:|---:|---:|
| FP16 | ✓ | ? | ? |
| INT8 | ? | ✓ | ? |
| INT4 | ? | ? | ✓ |

Now you're measuring **cross-precision refusal-vector transferability**.

That is much more research-like than simply measuring refusal rate.

---

# And another: which layers are damaged first?

Quantization probably won't perturb every part of the network identically.

Suppose the FP16 model has strongest harmful/benign separation around layers 14–18.

After INT4 maybe you see:

```text
             layer
         8  12  16  20  24

FP16     ▂   ▅   █   ▇   ▃
INT8     ▂   ▅   █   ▆   ▃
INT4     ▂   ▃   ▅   ▃   ▂
```

Then you can ask:

> **Does quantization disproportionately damage layers responsible for safety decisions?**

That becomes a much stronger mechanistic claim.

Recent work already suggests that models can differ in *where* safety decisions emerge—for example, relatively earlier versus later differentiation across model families. [arXiv](https://arxiv.org/abs/2606.22686?utm_source=chatgpt.com)

So maybe:

\[
\text{Qwen safety representation}
\]

is more robust to quantization than

\[
\text{Llama safety representation}
\]

—or the reverse.

That's publishable-style comparative research if the effect is clean.

---

# Your final experiment matrix

I would aim roughly for this:

| Dimension | Conditions |
|---|---|
| Model | 2 model families |
| Safety condition | Original / refusal suppressed / refusal enhanced |
| Precision | FP16 / INT8 / INT4 |
| Intervention order | Quantize→intervene / intervene→quantize |
| Prompt class | Harmful / benign / benign-risky |
| Evaluation | Behavioral + representational + utility |

That's already quite a lot.

Don't add ten quantization techniques initially.

Do:

\[
FP16,\;8\text{-bit},\;4\text{-bit}
\]

using **one consistent PTQ method**.

If you get a strong result, add another quantizer as an ablation study.

---

# What you should measure

Your paper should not have only “attack success rate”.

You want three families of measurements.

**Behavioral safety**

\[
\text{Unsafe compliance rate}
\]

\[
\text{Refusal rate}
\]

\[
\text{Over-refusal rate}
\]

StrongREJECT is particularly useful because it was designed to avoid falsely counting useless jailbreak responses as successful harmful responses. [arXiv](https://arxiv.org/abs/2402.10260?utm_source=chatgpt.com)

**Internal representation**

\[
\cos(r_{FP16},r_{INT4})
\]

layer-wise projection magnitude,

linear-probe accuracy,

harmful-vs-benign separation,

subspace dimensionality.

**General capability**

Something lightweight such as:

\[
\text{IFEval}
\]

plus one reasoning/knowledge benchmark.

This matters because otherwise you may observe:

> safety collapsed!

when the actual explanation is:

> the 2-bit model became terrible at everything.

The safety/utility distinction is particularly important because prior quantization work has found that ordinary metrics such as perplexity need not adequately capture alignment degradation. [arXiv](https://arxiv.org/abs/2511.07842?utm_source=chatgpt.com)

---

# The strongest hypothesis set

If I were advising the project, I'd probably write your proposal around four hypotheses:

\[
H_1:
\]

**Quantization changes refusal representations even when general task performance changes only slightly.**

\[
H_2:
\]

**Lower-bit quantization decreases similarity between the original and quantized refusal subspaces.**

\[
H_3:
\]

**Refusal-vector drift predicts behavioral safety degradation better than generic metrics such as perplexity.**

\[
H_4:
\]

**Safety intervention and quantization are non-commutative:**

\[
Q(A(M)) \neq A(Q(M)).
\]

That fourth one is particularly clean.

---

# What I would subtract from your original idea

I'd **not** spend half the project inventing jailbreak prompts.

Use existing benchmarks.

I'd also **not** make “creating an uncensored/offensive model” the research contribution. That's already very well explored and makes the project sound less sophisticated than it actually is.

And I wouldn't evaluate only refusal.

A model can be safe because:

1. it recognizes malicious intent and intentionally refuses, or
2. it became too degraded to provide useful answers.

Those are very different.

The internal activation measurements help separate them.

---

# What I would add

One particularly powerful addition would be a **representation-level safety predictor**.

Imagine discovering that before generating anything, you can compute:

\[
s(x)=h_l(x)\cdot r_l
\]

and this number predicts whether a quantized model will subsequently behave safely.

Representation-level safety metrics are becoming an active area of research; for example, recent SafeVec/RAS work explicitly evaluates safety from refusal-aligned hidden-state representations rather than outputs alone. [arXiv](https://arxiv.org/abs/2606.25750?utm_source=chatgpt.com)

So your project could ask:

> Can we detect quantization-induced safety degradation directly from internal representations without running a full expensive behavioral safety benchmark?

That adds a defensive contribution.

---

# The paper story could become extremely clean

Your introduction essentially becomes:

> Open-source LLMs are increasingly compressed for deployment. Prior work shows both that safety refusal has identifiable internal representations and that quantization can degrade safety. However, it remains unclear how quantization transforms the internal representations responsible for refusal and whether mechanistic safety interventions remain stable across precision levels. We study the geometry, transferability and intervention robustness of refusal representations under quantization.

Then your contribution statement becomes something like:

> We characterize layer-wise refusal representations before and after quantization, measure their cross-precision transferability, test whether quantization and refusal interventions commute, and relate representation drift to behavioral safety and over-refusal.

**That is a much better undergraduate research project than “we tested an abliterated model and then quantized it.”**

And crucially, it gives you several possible outcomes that are interesting.

If refusal vectors barely move under quantization → interesting.

If they rotate dramatically → interesting.

If geometry moves but safety doesn't → interesting.

If safety changes without geometry moving → also interesting, because it undermines the simple refusal-direction model.

If Qwen and Llama behave differently → interesting.

So you're not dependent on getting one predetermined result.

### The version I'd pursue

**Core project:**

> **How Stable Are Refusal Representations Under Quantization? A Mechanistic Study of Safety Alignment in Compressed LLMs**

with three main experiments:

\[
\boxed{\text{Representation drift}}
\]

\[
\boxed{\text{Cross-precision transferability}}
\]

\[
\boxed{Q(A(M)) \text{ versus } A(Q(M))}
\]

and behavioral safety + over-refusal + utility as your validation.

That scope is ambitious enough to produce an actual finding, but still realistic for an undergraduate team with modest GPUs. It also sits nicely at the intersection of **mechanistic interpretability, model compression, and alignment robustness**, rather than being another generic jailbreak project.