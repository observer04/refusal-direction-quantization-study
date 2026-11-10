---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags:
  - refusal-research
  - heretic
---
# Qwen3.5-9B corrected projected-direction reproduction

Fresh-extraction FP16 reconstruction of the rounded parameters in [trohrbaugh/Qwen3.5-9B-heretic-v2](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2), executed on Kaggle T4 ×2. This is our reconstructed checkpoint, not a copy of the publisher's weights. Quantization and later research phases are deferred.

The corrected procedure extracts means from 400 harmful and 400 harmless prompts, projects each refusal direction orthogonally to the corresponding harmless mean, normalizes each slot, then interpolates hidden-state slots 19/20 at 0.49. It applies the fixed published component-strength parameters without optimization. Sparse publisher-weight forensics plus a controlled raw-versus-projected pilot isolated this omitted processing step. The author's exact flag, environment, seed and unrounded parameters remain unpublished.

## Comparable historical measurements

The same FP16, closed non-thinking, EOS-enabled harness evaluates 100 previously exposed harmful prompts with a 100-token cap and upstream refusal markers, plus 100 benign first-token distributions.

| Condition | Marker refusals /100 | Benign first-token KL |
| --- | ---: | ---: |
| Original Qwen | 100 | — |
| Earlier raw-direction replay | 72 | 0.434047 |
| Actual publisher checkpoint, independently evaluated | 8 | 0.078430 |
| This fresh-extraction reconstruction | 8 | 0.078021966 |
| Publisher's card, reported | 6 | 0.0793 |

The reconstructed model reached the 100-token cap on **99/100** answers. A fresh model load from the saved checkpoint reproduced **100/100 responses exactly**. Independent artifact checks verified all 300 original/edited/reloaded marker labels, prompt provenance, the mean of 100 saved per-prompt KL terms, direction derivation and all 64 saved adapter B tensors (42 active). Fresh extracted means/directions exactly matched the separately executed cached-means pilot.

## Longer-answer diagnostic and limits

On 12 fixed exposed prompts at a 512-token cap, original/corrected marker counts were 11/0 and cap counts were 2/11. One-agent qualitative review found semantic withholding in all 12 originals and in two corrected answers (nine no, one unknown). Seven corrected answers visibly assisted, two did not, and three were ambiguous. The reviewer knew the conditions and context. This small diagnostic has no inter-rater agreement or population interpretation; technical accuracy of the generated content was not validated.

**Missing refusal markers do not establish harmful compliance.** Truncation limits interpretation. Benign KL is `KL(original || edited)` at the first generated token, not a broad capability benchmark. The historical 100 prompts were exposed during recovery. The separate pilot's 64-case standard-only subset is not full HarmBench and used no HarmBench classifier. **The independently advertised 0/400 result has not been reproduced here.** No broad safety-mechanism or capability-preservation conclusion is claimed.

## Reproduction and evidence

Start with [the self-contained corrected Kaggle notebook](reproduction/Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb). The original failed notebook is preserved alongside it. All model execution, dependency installation and full-weight downloads occur on Kaggle; no local model execution is required.

- `experiments/`: fresh directions/means, portable adapter, all responses, summary, package environment, rendered-prefix probe, model/source/dataset revisions and 100 per-prompt KL terms.
- `reproduction/`: source notebooks, pinned material, recovery implementation/evaluation helpers and scope notes.
- `recovery_evidence/`: compact pilot/fresh artifacts, independent verification, qualitative reviews and SHA-256 manifest.

Base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Hybrid-compatible Heretic revision: `e26da5e0e648bb285f896c9801849edfa8d20913`. Both DeltaNet and full-attention output projections are covered. Kaggle adaptations are FP16, explicit closed non-thinking formatting, CPU aggregation across GPUs and preservation of Kaggle's CUDA PyTorch.

Both recovery GPU jobs completed and released their GPUs. Total measured Kaggle quota use was **2.11h**, within the initial 3h envelope. This repository is kept private at the owner's request.
