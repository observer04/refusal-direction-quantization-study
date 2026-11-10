# Understanding Guide: "Does Abliteration Survive 4-bit Quantization?"

This guide is for the three of us presenting the report. It assumes you know basic Python and have heard of neural networks, but nothing more. Read Part 1 first (10 minutes). Parts 2 to 4 explain everything in the report from zero. Part 5 decodes every equation with a tiny worked example. Part 6 is a list of likely examiner questions with answers. Part 7 lists things we must **not** claim.

Report section numbers are given in brackets, for example [3.2], so you can jump between this guide and the report.

---

## Part 1. The whole project in one page

**The setting.** Chat models such as Qwen are trained to refuse harmful requests ("I can't help with that"). In 2024, researchers found that this refusal behavior is controlled mostly by **one direction** inside the model's internal numbers. If you delete that direction from the model's weights, the model mostly stops refusing. People call this edit **abliteration**, and there are tools (one is called **Heretic**) that do it automatically.

**The problem.** People make the abliteration edit on the full-size model (16-bit numbers). But most people then **shrink** the model to 4-bit numbers (quantization) so it fits on a normal GPU. Shrinking rounds every weight. Nobody checks whether the edit survives that rounding.

**Our question.** *If you abliterate a model and then quantize it to 4-bit (NF4), does the edit still work? And if something changes, is it because the edit was lost, or just because quantization damages the model in general?*

**How we answered it, in three steps:**

1. **Get a faithful edit (SQ1).** We rebuilt a published Heretic edit of Qwen3.5-9B. Our first attempt failed: 72 of 100 harmful prompts still produced refusal phrases. We found the missing step, which is to subtract the "harmless average" part from the refusal direction ("projected abliteration"). After fixing it we got 8 of 100, the same as the publisher's model.
2. **Get a trustworthy measure (SQ2).** Counting refusal phrases ("sorry", "I can't") is misleading. A model can say "sorry" and then answer anyway, or avoid refusal words and still not answer. So we judged what each answer actually contained: clear answer, partial answer, or no answer.
3. **Test survival (SQ3).** We compared five versions of the model (original, edited, and their 4-bit versions, plus one extra). The edit's effect was still there after 4-bit quantization, and quantizing the *original* model did not create the effect by itself.

**Bottom line.** On the prompts we tested, the edit survived 4-bit quantization. The hard part turned out to be measuring correctly, not the computing. The tests were small, so we cannot rule out that the edit got *a bit* weaker.

If you remember only one sentence: **"The abliteration edit kept its effect after 4-bit quantization on our tests, but to show that we first had to fix the reproduction and stop trusting refusal-phrase counts."**

---

## Part 2. Background concepts from zero

### 2.1 What a language model does
A language model reads text as **tokens** (pieces of words) and predicts the next token, one at a time. Each token becomes a long list of numbers (a **vector**). For Qwen3.5-9B each vector has **4096 numbers**. This vector goes through **32 layers** (also called blocks). Each layer reads the vector and adds some changes to it.

### 2.2 The residual stream [2.1]
Think of the 4096-number vector as a **shared whiteboard**. Each layer reads the whiteboard, computes something, and **adds** its result onto the whiteboard. This whiteboard is called the **residual stream**.

Each layer writes onto the whiteboard through two "pens", which are weight matrices:
- the **attention output projection** (from the attention part of the layer), and
- the **MLP down projection** (from the feed-forward part).

Key idea: if you want to stop *all* layers from writing a certain thing onto the whiteboard, you only need to change these output matrices. That is exactly what abliteration does.

### 2.3 Vectors, directions and projections (the only math you really need)
A **direction** is just an arrow in the 4096-dimensional space. The **dot product** of a vector with a unit-length direction tells you "how much of the vector points along that direction".

Tiny 2-D example. Vector y = (2, 5). Direction r = (0, 1), which points straight up and has length 1.
- How much of y points along r? The dot product is r · y = 0×2 + 1×5 = **5**.
- Remove that part: y − 5·r = (2, 5) − (0, 5) = **(2, 0)**. Now y has nothing pointing up.

That "remove the part along a direction" operation is called **projecting out** a direction. It is the heart of abliteration.

### 2.4 The refusal direction [2.2]
How do we find the direction that means "refuse"?
1. Feed the model 400 **harmful** prompts and record the residual-stream vector (at one layer, at the last token of the prompt). Average them. That is the **harmful mean**.
2. Do the same with 400 **harmless** prompts. That is the **harmless mean**.
3. Subtract: harmful mean − harmless mean. This difference points from "normal request" toward "harmful request", and the model uses it to decide to refuse. This is the **refusal direction** (after scaling it to length 1).

Arditi et al. (reference [1] in the report) showed that removing this one direction stops refusal in many chat models, and adding it makes models refuse harmless requests.

### 2.5 Abliteration [2.2, 3.3]
Instead of removing the direction while the model runs, abliteration **changes the weights permanently**, so the layers can no longer write along the refusal direction. The edited model can be saved and shared like any other model.

There is a **strength** called λ (lambda):
- λ = 1: remove the part along the direction completely.
- λ less than 1: remove only some of it.
- λ greater than 1: overshoot, so the part is *reversed* (pushes away from refusal).

In the 2-D example with λ = 2: y − 2·5·r = (2, 5) − (0, 10) = (2, −5). The "up" part flipped sign.

### 2.6 Heretic [2.2, 3.3]
Heretic is an open-source tool that does abliteration automatically. Two features matter for us:
- It gives **each layer its own strength** λ. The strength is highest at one "peak" layer and falls off in a straight line with distance from it, becoming zero beyond a cutoff (Fig. 3.1 shows this as bars).
- It **searches** for the best settings, trading off "fewer refusals" against "model changed as little as possible" (measured by KL divergence, see 2.9).

We did **not** run the search. We used the settings the publisher released.

### 2.7 Projected abliteration: the fix we needed [2.2, 4.1]
The raw difference (harmful mean − harmless mean) contains some of the **harmless mean** direction itself. That part is shared by *every* prompt, so removing it disturbs the model on all inputs, not just harmful ones. A 2025 community article proposed removing that shared part first ("projected abliteration"). Heretic does this, and our first replay forgot to. Fixing it changed our results from 72 to 8 refusal-phrase matches out of 100. That is our SQ1 story.

### 2.8 The model: Qwen3.5-9B [2.1, 3.1]
- 9 billion parameters, 32 layers, vectors of 4096 numbers.
- It is a **hybrid**: 24 layers use **Gated DeltaNet** (a cheaper "linear attention" with a memory state) and 8 use normal **full attention**, in a repeating pattern of 3 linear then 1 full.
- Both kinds of layers end in an output projection, so abliteration works the same way on both.
- "Published edit P" = the model `trohrbaugh/Qwen3.5-9B-heretic-v2`, which someone made with Heretic and uploaded.

### 2.9 KL divergence [3.6]
KL divergence measures **how different two probability distributions are**. We use it to check how much the edit changed the model on *harmless* prompts. For each harmless prompt, we look at the model's probabilities for the **first word** of its answer, before and after the edit. KL = 0 means no change. Bigger means more change. We average over 100 harmless prompts. Our edited model had KL 0.078 (small). The broken replay had 0.434 (much bigger). See Part 5 for a worked example.

### 2.10 Quantization and NF4 [2.5, 3.4]
Normally each weight is stored as a 16-bit number (**FP16**). **Quantization** stores weights with fewer bits. With 4 bits there are only 16 possible values, so every weight gets **rounded** to the nearest allowed value. That makes the model about 4 times smaller.

**NF4** ("4-bit NormalFloat", from the QLoRA paper [3]) works like this:
1. Split weights into **blocks of 64**.
2. In each block, divide by the largest absolute value (the **scale**), so all weights are between −1 and 1.
3. Round each weight to the nearest of 16 fixed levels. The levels are packed more tightly near 0, because most weights are small (they follow roughly a bell curve, a "normal" distribution, hence the name).
4. Store the 4-bit level numbers plus the one scale per block. When computing, multiply back to get approximate weights.

Worked example in Part 5.

### 2.11 FP16, FP32, LoRA, rank one
- **FP16 / FP32**: 16-bit and 32-bit floating-point numbers. FP32 is more precise.
- **Rank one**: the abliteration change to a matrix can be written as (a column vector) × (a row vector). That is called a rank-one matrix. It can be stored very compactly as those two vectors.
- **LoRA adapter**: a common way to store small low-rank changes to a model separately from the main weights. Our edit is stored in that format (one pair of vectors per matrix) and then **merged** (added) into the weights.

### 2.12 The prompt datasets [3.5]
- **mlabonne/harmful_behaviors** and **mlabonne/harmless_alpaca**: harmful and harmless prompts for finding the direction (400 each) and for the 100-prompt test.
- **HarmBench**: a standard benchmark of harmful requests. We used a few as safety prompts.
- **XSTest**: *safe* prompts that sound unsafe (for example "how do I kill a Python process"). They check **over-refusal**, meaning refusing things that are fine.
- **GSM8K**: grade-school math word problems. A basic check that the model can still reason.

---

## Part 3. The report, chapter by chapter

### Chapter 1. Introduction
- **What it says:** abliteration exists, people quantize models afterwards, nobody checks whether the edit survives. It states the research question and splits it into SQ1, SQ2 and SQ3. It lists three contributions, and it has an ethics section.
- **What you should be able to say:** the question in one sentence, and why each sub-question is needed: "you can't test survival if your edit is wrong (SQ1) or your measure is wrong (SQ2)".

### Chapter 2. Background and related work
- [2.1] The residual stream (Part 2.2 above) and Qwen's hybrid layers.
- [2.2] How the refusal direction is found (Arditi), how abliteration and Heretic work, and projected abliteration.
- [2.3] Refusal is probably *not* exactly one direction. Other papers found several related directions. So we must check answers, not just the geometry.
- [2.4] How to measure refusal. HarmBench uses a trained classifier. StrongREJECT says "judge whether the answer is actually useful, not whether it avoided refusing". XSTest checks over-refusal. Abliterlitics is a community site that claims the published model refused 0 of 400 prompts.
- [2.5] Quantization and safety. Other papers show quantization can change how safe a model is (Q-resafe, Kumar, Kadadekar), or can even be used as a trigger to hide bad behavior (Egashira). All of them study the model's *original* safety training, not a deliberate edit added afterwards. That is our gap.
- [2.6] Table 2.1 summarizes the papers and states the gap: abliteration papers test only full precision, and quantization papers test only the original training.

### Chapter 3. Methodology
- [3.1] W = original Qwen3.5-9B, P = published edited model. We rebuilt the edit ourselves with Heretic's code and the published settings, on 2 Kaggle T4 GPUs.
- [3.2] How the direction is computed (equation 3.1, decoded in Part 5). The setting "direction index 18.49" means "use a mix of the directions from layers 18 and 19, with 49% weight on the second".
- [3.3] How the weights are edited (equations 3.2 and 3.3). The strength per layer and which layers get edited are shown in Fig. 3.1: 19 attention matrices (layers 10 to 28) and 23 MLP matrices (layers 9 to 31), so **42 edited matrices**. Δ (delta) is the name for the whole change, and D = W + Δ is our edited model. Algorithm 3.1 lists the steps.
- [3.4] The five conditions (Table 3.1, Fig. 3.2). See Part 4.5 below, which is the most important design idea.
- [3.5] Prompts and generation settings (Table 3.2). "Greedy decoding" means the model always picks its single most likely next token, so results are repeatable. "Token limit" means answers were cut off after that many tokens.
- [3.6] The measures: phrase matching, KL, semantic review, and the AI-assistance statement.

### Chapter 4. Results
- [4.1] SQ1: Table 4.1 and Fig. 4.1. The fix took refusal phrases from 72 to 8 and KL from 0.434 to 0.078. Weight checks passed. Save and reload gave identical outputs.
- [4.2] SQ2: Tables 4.2 and 4.3, Fig. 4.2. Phrase counts and semantic labels disagree.
- [4.3] SQ3: Table 4.4, Fig. 4.3. The effect survived NF4.
- [4.4] Why it might survive: our hypothesis, not tested.
- [4.5] Limitations, plus Fig. 4.4 (the claim ladder).

### Chapter 5. Conclusion and future scope
The answer, the two lessons (reproduction and measurement), and three next steps ranked by priority.

### Appendix A
Sources (Table A.1), software versions and GPU hours, and the bounds calculation (equation A.1).

---

## Part 4. How to read every table and figure

### Table 2.1 (related work)
Each row: a paper, what it found, and how we used it. The last row is our gap.

### Figure 3.1 (where the edit is applied)
- Top strip: 32 layers. Black = full attention, grey = linear attention.
- Bars: the strength λ for each layer. Blue = attention output, orange = MLP output. Zero bars mean "not edited".
- Takeaway: the edit is spread over the middle and late layers, and many strengths are above 1 (overshooting).

### Algorithm 3.1
Seven plain steps: compute the means, compute the direction, compute the strengths, edit, verify, save and reload.

### Table 3.1 and Figure 3.2 (the five conditions): the key design idea
| Label | What it is | Why we need it |
| --- | --- | --- |
| W | original, 16-bit | the "before" |
| D = W + Δ | our edited model, 16-bit | the "after" at full precision |
| Q(W) | original, 4-bit | what 4-bit does to a model *without* the edit |
| Q(W + Δ) | edited, then 4-bit | what people actually download |
| Q(W) + Δ | original 4-bit, with the edit added on top in 32-bit | the edit itself is never rounded |

(P, the publisher's model, is also in Table 3.1 but is only used in the 16-bit comparisons.)

The logic:
- **Edit effect at full precision** = behavior of D minus behavior of W.
- **Edit effect after 4-bit** = behavior of Q(W + Δ) minus behavior of Q(W).
- If those two are about equal, the edit survived.
- **Why Q(W) matters:** without it, if Q(W + Δ) behaved differently we could not tell whether the edit was lost or 4-bit just damaged the whole model. Q(W) is the **control**.

### Table 3.2 (prompt sets)
"Matched" = the 6 safety prompts reused from the pilot. "Fresh" = 8 new safety prompts, chosen and fixed before we read any answers. Token limits are given for safety / safe / arithmetic prompts.

### Table 4.1 (reproduction)
- **Marker matches**: how many of the 100 answers contained a refusal phrase.
- **Reached limit**: how many answers were cut off at 100 tokens. 99 of D's answers were cut off, so we **cannot** say "92 answered".
- **Benign KL**: how much the edit changed the model on harmless prompts.
- Takeaway: the raw replay is broken (72, KL 0.434). The fixed version matches the publisher (8, KL 0.078).

### Figure 4.1
Left: a 2-D sketch of the raw direction d versus the projected direction r. Right: the same two numbers as bars.

### Table 4.2 (semantic review)
- Clear / Partial / None: did the answer actually give the requested content?
- Explicit refusal: did it contain refusal language? (This can be "yes" even when the answer is clear, which is the point.)
- Takeaway: on 8 fresh prompts, the original clearly answered 1 and the edited models 6. Our model and the publisher's got identical labels.

### Table 4.3 and Figure 4.2 (three measures disagree)
For the same answers: phrase-matching count, the Abliterlitics v3 classifier count, and our clear-fulfillment count. The v3 classifier says 0 refusals for the edited models on fresh prompts, yet 2 of those 8 answers were only partial. So "0 refusals" does not mean "everything answered".

### Table 4.4 and Figure 4.3 (NF4 pilot)
- W and Q(W) are identical (0 clear), so 4-bit alone creates nothing.
- D, Q(W + Δ) and Q(W) + Δ each have 2 clear, so the edit's visible effect survived.
- The hatched "unknown" labels and the many cut-off answers are why we cannot be more precise.

### Figure 4.4 (the claim ladder)
Four claims, each needing stronger evidence than the one below:
1. The edit is implemented correctly. **Established.**
2. Refusal *wording* disappears. **Established.**
3. The requested *content* is supplied. **Evidence on selected cases only, from one reviewer.**
4. The content is *correct*. **Deliberately not tested.**
This figure is your best friend in Q&A when someone overreaches.

---

## Part 5. Every equation, decoded with a toy example

You do not need to derive anything. You need to say in words what each equation does.

### Equation 3.1: the projected refusal direction
r = (d − (d·u) u) / ‖d − (d·u) u‖

- d = harmful mean − harmless mean (the raw direction).
- u = the harmless mean scaled to length 1.
- d·u = how much of d points along the harmless mean.
- d − (d·u)u = d with that part removed.
- Dividing by ‖…‖ (the length) makes r have length 1.

**In words:** "take the raw difference, remove the part that points along the harmless average, and scale to length one."

Toy example: d = (3, 4), u = (1, 0). Then d·u = 3, so d − 3u = (0, 4), whose length is 4, so r = (0, 1).

### Equation 3.2: the weight edit
M' = M − λ r rᵀ M

- M = a layer's output matrix (its "pen" onto the whiteboard).
- rᵀ M = for every input, how much this layer writes along r.
- r rᵀ M = that amount, pointing along r.
- Subtract λ times it.

**In words:** "make the layer write λ times less along the refusal direction." With λ = 1 it can write nothing along r at all. Toy example: Part 2.3 and 2.5 above (y = (2, 5) becomes (2, 0) with λ = 1, or (2, −5) with λ = 2).

### Equation 3.3: strength per layer (Heretic's kernel)
λ for layer l = λmax + (|l − p| / w)(λmin − λmax) if |l − p| ≤ w, otherwise 0

- p = peak layer, w = width, λmax = strength at the peak, λmin = strength at the edge of the window.
- **In words:** "the strength is highest at the peak layer, falls in a straight line to λmin at distance w, and is zero beyond that."
- Attention example: p = 18.99, w = 9.65, λmax = 2.00, λmin = 1.43. Layer 19 is about 0.01 from the peak, so λ ≈ 2.0. Layer 10 is 8.99 away, so λ = 2.00 + (8.99/9.65)(1.43 − 2.00) ≈ 2.00 − 0.53 = 1.47. Layer 9 is 9.99 away, which is more than 9.65, so λ = 0 (not edited). That is why the attention edit covers layers 10 to 28.

### Equation 3.4: KL divergence
KL = (1/N) Σᵢ Σₜ p_W(t | xᵢ) log( p_W(t | xᵢ) / p_D(t | xᵢ) )

- For each harmless prompt xᵢ, compare the original model's probability of each possible first word t, p_W, with the edited model's, p_D.
- Average over N = 100 prompts.

Toy example with three possible first words:
| word | original p_W | edited p_D |
| --- | --- | --- |
| "Sure" | 0.6 | 0.5 |
| "I" | 0.3 | 0.4 |
| "The" | 0.1 | 0.1 |

KL = 0.6·ln(0.6/0.5) + 0.3·ln(0.3/0.4) + 0.1·ln(1) = 0.109 − 0.086 + 0 ≈ **0.023**. Small, because the distributions are similar. Identical distributions give 0.

### NF4 rounding (no numbered equation, but you should be able to do this)
NF4's 16 levels are approximately: −1, −0.696, −0.525, −0.395, −0.284, −0.185, −0.091, 0, 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.

Block of four weights (real blocks have 64): [0.8, −0.3, 0.05, −1.6]
1. Scale = largest absolute value = 1.6. Divide: [0.5, −0.19, 0.03, −1.0].
2. Round to the nearest level: [0.441, −0.185, 0, −1].
3. Multiply back by 1.6: [0.706, −0.296, 0, −1.6].
4. Errors: [0.094, 0.004, 0.05, 0]. Small but not zero. That rounding error is what could, in principle, erase the edit.

### Equation 4.1: the retained fraction ρ (rho), proposed future test
ρ = ⟨Q(W+Δ) − Q(W), Δ⟩ / ‖Δ‖²

- Q(W+Δ) − Q(W) = the change you actually get between the two 4-bit models.
- Δ = the change you intended.
- ⟨a, b⟩ = multiply matching entries and add them up (a dot product over matrix entries).
- ‖Δ‖² = ⟨Δ, Δ⟩.

**In words:** "what fraction of the intended edit is still there after quantization". ρ ≈ 1 means the edit fully survived. ρ ≈ 0 means it was rounded away.

Toy example: intended Δ = [1, 2]. Actual difference = [0.9, 2.1]. ⟨actual, Δ⟩ = 0.9 + 4.2 = 5.1. ‖Δ‖² = 1 + 4 = 5. So ρ = 5.1 / 5 = **1.02**, which means essentially all of it survived.

### Equation A.1: the interaction I (the bounds in Section 4.3)
I = [H(Q(W+Δ)) − H(Q(W))] − [H(D) − H(W)]

- H(·) = fraction of the 6 safety prompts that a condition clearly answered.
- First bracket = edit effect after 4-bit. Second bracket = edit effect at full precision.
- **I = 0 means the edit survived unchanged.**

With our labels: [2/6 − 0/6] − [2/6 − 0/6] = 0.

**The bounds −0.50 to +0.67:** some answers were labeled "partial" or "unknown". If we pretend each of those could be either "answered" or "not answered", I could range from −0.50 to +0.67. If we also imagine that cut-off answers might have turned into full answers, the range becomes −0.83 to +0.83. These are **not** statistical confidence intervals. They just show how much the uncertain labels could matter.

### The "why it survives" argument [4.4], in plain words
The edit is like **one loud, coordinated push** in a single direction. Rounding error is like **lots of tiny random nudges** spread across all 4096 directions. If you look along the single refusal direction, the random nudges mostly cancel out and stay small, but the coordinated push is still there in full. So the edit should survive rounding unless the push is tiny or the rounding is very coarse. This is a **hypothesis** that we did not test. Testing it is future work item 1 (compute ρ).

---

## Part 6. Likely examiner questions, with answers

**Basics**

1. **What is abliteration?** A permanent weight edit that removes the "refusal direction" from a model, so the model's layers can no longer write that direction into its internal state. As a result the model mostly stops refusing.
2. **How is that different from fine-tuning?** Fine-tuning trains the model on new data with gradient descent. Abliteration needs no training. It computes one direction from averages and edits the output matrices with a fixed formula.
3. **What is the refusal direction, concretely?** The difference between the average internal vector for harmful prompts and the average for harmless prompts, at a chosen layer, scaled to length one.
4. **What is quantization, and why do people do it?** Storing weights with fewer bits (here 4 instead of 16) by rounding them. The model becomes about 4 times smaller, so it fits on smaller GPUs.
5. **What is NF4 specifically?** A 4-bit format with 16 levels spaced to match bell-shaped weight distributions, applied in blocks of 64 weights, each with its own scale. It comes from the QLoRA paper.

**Design**

6. **Why Qwen3.5-9B?** A published Heretic edit of it existed, so we had a reference to reproduce. It is a recent hybrid architecture, and it fits in FP16 across two free Kaggle T4 GPUs.
7. **Why not just use the publisher's model?** We needed to know exactly what the edit Δ was, to build Q(W) + Δ and to verify the edit matrix by matrix. Reproducing it also checks that the method works as published.
8. **Why is the Q(W) condition important?** It is the control. It shows what 4-bit does to a model with no edit. Without it we could not separate "edit lost" from "4-bit damaged everything".
9. **What is Q(W) + Δ for?** It keeps the edit in 32-bit on top of the 4-bit base, so the edit itself is never rounded. If Q(W + Δ) had lost the effect but Q(W) + Δ kept it, that would point to rounding of the edit as the cause.
10. **Why NF4 and not GPTQ, AWQ or GGUF?** NF4 is simple, has no calibration step, and is built into the Hugging Face stack we used. Other formats are future work.
11. **Why greedy decoding?** It makes outputs deterministic and repeatable, so differences come from the model and not from random sampling.

**Results**

12. **Why did your first replay fail?** It skipped the projection step that removes the harmless-mean part of the direction. With the same settings otherwise, refusal-phrase matches went from 72 to 8 out of 100 once we added it, and KL fell from 0.434 to 0.078.
13. **Why does that projection help?** The raw direction partly points along the harmless average, which every prompt shares. Removing that shared part disturbs all prompts (higher KL), and it wastes part of the edit on something that is not refusal (weaker effect).
14. **8 of 100 refusals means 92% success, right?** No. 99 of the 100 answers were cut off at 100 tokens, and a missing refusal phrase does not mean the request was answered. That is exactly why we did the semantic review.
15. **How do you know your edit matches the publisher's?** Same refusal count (8) and nearly the same KL (0.078 for both) on the same test. Identical fulfillment labels on all 14 safety prompts. 8 of 24 paired answers were byte-identical. Their weight changes are in the same matrices and point in the same direction (cosine above 0.9999 where we compared).
16. **What does "survive" mean in your result?** The two prompts the edit clearly answered at full precision were still clearly answered after 4-bit, while the 4-bit original answered none. So the edit effect was 2 of 6 before and after.
17. **Can you say the edit is 100% preserved?** No. With 6 prompts and several unclear or cut-off answers, a partial weakening could go undetected. We can say we saw no erasure.
18. **What is the Abliterlitics 0/400 claim, and do you agree?** A community site reports that the published model refused 0 of 400 prompts using its own classifier. We could not replicate it: they used different settings, longer outputs and manual overrides, and their raw answers were not available. We also showed their classifier counts a polite redirect as non-refusal, and we found a bug in how it handles reasoning tags. We do not claim that bug explains their number.

**Limitations and honesty**

19. **Who labeled the answers?** A single AI (LLM-based) reviewer. It saw the answers shuffled and without condition names, and fixed its labels before the mapping was revealed. It knew some earlier results, so the blinding was imperfect. No humans have labeled them yet. That is a limitation, and human review is future work item 2.
20. **Did you check whether the harmful answers were correct?** No, deliberately. That would require judging and handling harmful content in detail, which we avoided for ethical reasons. Our claim stops at "content was supplied".
21. **Why so few prompts?** Free GPU quota (Kaggle) and long generations. The work used about 4 GPU-hours across the three main runs. We treat the results as a pilot.
22. **Is this research ethical?** We reproduced only an edit that was already public, quote no harmful output, keep raw outputs private, and do not release the weights. Understanding whether safety edits survive deployment matters equally for edits that *add* safety.
23. **What did AI tools do in this project?** AI coding agents helped with implementation, running the experiments and labeling, as disclosed in Section 3.6. *(Answer this one honestly in your own words, based on what each of us actually did.)*

**Big picture**

24. **What is new here?** Not a new method. The new part is a controlled test of whether a post-training safety edit survives quantization, with a control that separates edit loss from general damage, and a demonstration that common refusal metrics can mislead.
25. **How does this differ from Q-resafe and similar papers?** They study what quantization does to the model's *original* safety training. We study a *deliberate edit added afterwards*.
26. **Is refusal really one direction?** Probably not exactly. Other papers found several related directions. That is why we checked the model's actual answers instead of trusting the geometry.
27. **What would you do next?** (1) Compute ρ on the weights for several bit widths, which is cheap and needs no text generation. (2) Get human labels. (3) Run a bigger study with more prompts, more quantizers and another model.
28. **What does λ greater than 1 mean?** Overshoot: the layer's output along the refusal direction is flipped, not just removed. Heretic's search chose values up to 2.0.

---

## Part 7. Things we must NOT claim

- Do **not** say "the model answers 92% of harmful prompts". We never measured that.
- Do **not** say "quantization never affects abliteration". It is one model, one quantizer and 6 prompts.
- Do **not** say the harmful answers were accurate or dangerous. We did not check.
- Do **not** say we proved why the edit survives. Section 4.4 is a hypothesis.
- Do **not** say we invented abliteration or projected abliteration. We reproduced and tested them.
- Do **not** say the Abliterlitics result is wrong or fraudulent. We only say we could not replicate it and that their classifier can miss some cases.
- Do **not** say humans reviewed the answers. An AI reviewer did.

---

## Numbers cheat sheet

| What | Number |
| --- | --- |
| Model size / layers / vector size | 9B / 32 layers / 4096 |
| Layer mix | 24 Gated DeltaNet (linear) + 8 full attention, pattern 3:1 |
| Prompts used to find the direction | 400 harmful + 400 harmless |
| Edited matrices | 42 (19 attention, layers 10 to 28; 23 MLP, layers 9 to 31) |
| Strength range | 0.84 to 2.00 |
| Refusal phrases, 100-prompt test | original 100, broken replay 72, ours 8, publisher 8 (publisher's own card says 6) |
| Benign KL | broken 0.434, ours 0.078, publisher 0.078 |
| Answers cut off at 100 tokens (ours) | 99 of 100 |
| Save and reload | 100 of 100 identical |
| Fresh safety prompts, clear answers | original 1 of 8, ours 6 of 8, publisher 6 of 8 |
| Matched safety prompts, ours and publisher | 2 clear, 3 partial, 1 none (of 6) |
| Identical paired answers, ours vs publisher | 8 of 24 |
| NF4 pilot, clear answers (of 6) | W 0, D 2, Q(W) 0, Q(W+Δ) 2, Q(W)+Δ 2 |
| Bounds on the change in edit effect | −0.50 to +0.67 (−0.83 to +0.83 with cut-off answers) |
| GPU time | about 4 hours on 2× T4 (2.11 + 1.04 + 0.87) |

## Glossary

- **Activation:** the numbers a layer produces for a token.
- **Ablation:** removing something (here, a direction) to see what changes.
- **Abliteration:** ablation + obliteration. A permanent weight edit that removes the refusal direction.
- **Control:** a condition that differs only in the thing you want to rule out (here, Q(W)).
- **Dot product:** multiply matching entries and add. Measures "how much one vector points along another".
- **Greedy decoding:** always pick the most likely next token.
- **Heretic:** a tool that automates abliteration and searches its settings.
- **KL divergence:** how different two probability distributions are; 0 means identical.
- **LoRA / rank one:** a compact way to store a small change to a matrix as a pair of vectors.
- **Marker / phrase matching:** counting answers that contain refusal words like "sorry".
- **NF4:** a 4-bit weight format with 16 levels per block of 64 weights.
- **Over-refusal:** refusing a safe request because it sounds unsafe.
- **Projection:** the part of a vector that points along a direction.
- **Quantization:** rounding weights to fewer bits to save memory.
- **Residual stream:** the shared vector that every layer reads from and adds to.
- **Semantic review:** judging what an answer actually says (clear, partial or no fulfillment).
- **Token limit / truncation:** answers were cut off after a fixed number of tokens.
