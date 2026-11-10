# Qwen3.5-9B Heretic reproduction

Start with [Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb](../../Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb). Upload this self-contained notebook to Kaggle, enable **GPU T4 ×2** and **Internet**, and run it in a fresh session. All dependencies, model downloads and model execution remain on Kaggle. Supply `HF_TOKEN` privately through the process, Kaggle Secrets or the owner's private runtime dataset.

The corrected notebook extracts fresh activation means from 400 harmful and 400 harmless training prompts, projects each refusal vector orthogonally to its corresponding harmless mean, normalizes each slot, then interpolates hidden-state slots 19/20 at 0.49. It applies the fixed published component weights; no parameter search or quantization. It evaluates 100 harmful prompts and 100 benign next-token distributions, saves an adapter and merged checkpoint, reloads the saved checkpoint and requires 100 exact response matches before private Hugging Face persistence. A fixed 12-prompt paired 512-token diagnostic compares the original and reloaded corrected model. Missing refusal phrases do not establish harmful compliance.

The fresh corrected run completed on Kaggle: **8/100 marker refusals, benign first-token KL 0.078021966, and 100/100 exact save/reload response matches**. The [full checkpoint is private on Hugging Face](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-corrected-20261109). Fresh means and projected directions exactly matched the separately executed cached-means pilot. Independent artifact checks verified prompt provenance, all 300 native response labels, the mean of 100 saved KL terms, direction derivation and all 64 saved adapter B tensors (42 active); A/assignment checks ran on Kaggle. Reload inference used the saved checkpoint before upload; remote persistence checks cover the index and shard content metadata. See [implementation evidence](recovery/IMPLEMENTATION_NOTES.md), [evaluation protocol](recovery/evaluation_protocol.md), and [current state](../../STATE.md).

## Why the procedure changed

The historical raw-direction replay gave 72/100 marker refusals and benign first-token KL 0.434047. Under the same FP16/non-thinking harness, the actual publisher checkpoint gave 8/100 and 0.078430, close to its card's reported 6/100 and 0.0793. Sparse weight-direction fingerprints from an MLP and a full-attention projection aligned almost perfectly with harmless-mean-orthogonalized directions; sampled target coverage also included DeltaNet projections. A controlled raw-versus-projected pilot then changed development refusals from 20/32 to 0/32 and KL from 0.437735 to 0.065908 while keeping component-strength parameters, prompts and extraction means fixed. Independent artifact checks verified saved vectors, adapter scaling and target masks. This supports the omitted projection step as the recovered discrepancy, while the author's exact settings remain unpublished.

The original [Reproduce_Qwen3_5_9B_Heretic.ipynb](../../Reproduce_Qwen3_5_9B_Heretic.ipynb), [historical results](PHASE1_RESULTS.md), and failed replay checkpoint remain preserved as evidence. Use the corrected entrypoint for the next reproduction.

## Pinned material

- [Publisher's card](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2): selected parameters and 100/100→6/100 marker refusals, KL 0.0793; pinned copy in `reference/published_model_card.md`.
- [Official Heretic source](https://github.com/p-e-w/heretic): the untouched release tag in `heretic-v1.2.0/` predates Qwen3.5 support and is retained for reference.
- Hybrid-compatible upstream source in `heretic-qwen35/`, revision `e26da5e0e648bb285f896c9801849edfa8d20913`, still version 1.2.0. The notebook downloads its archive and verifies SHA-256.
- `published_parameters.json`, `sources.json`, and `requirements-kaggle.txt` preserve values and source/model/dataset/package pins. These reconstruct our environment, not the author's unpublished environment.

Kaggle adaptations are explicit FP16, the closed non-thinking Qwen template, CPU aggregation across GPUs and removal of unused conflicting torchao while preserving Kaggle PyTorch. Both full-attention and DeltaNet output projections are covered. The corrected notebook explicitly applies harmless-mean projection once, before direction interpolation.

## Evaluation limits

The comparable historical test contains 100 prompts already exposed during recovery. The 100-token marker score can miss semantic refusals and truncate answers. Benign KL measures only the first-token distribution and is not a broad capability score. The pilot's 64-case fresh standard-only subset is not full HarmBench and uses no HarmBench classifier. The independent evaluator's 0/400 claim has not been reproduced. The fixed 12-prompt paired 512-token diagnostic had 11 original markers and zero corrected markers. One-agent qualitative review found semantic withholding in all 12 originals but still in two corrected answers; seven corrected answers visibly assisted, with three ambiguous. Original 2/12 and corrected 11/12 hit the cap. These exposed diagnostic cases and reviewer labels are retained separately; they do not estimate a population rate.

No author notebook, complete optimizer journal, exact environment/seed or unrounded parameters was found. This is an approximate measured FP16 reconstruction, not byte-identical author weights. Quantization and later research phases remain deferred.

## Bounded batch execution

The corrected private metadata is in `recovery/kaggle-corrected/`. Submit with `kaggle kernels push -p reproduction/qwen35_9b/recovery/kaggle-corrected --accelerator NvidiaTeslaT4 --timeout 6300`. Its runtime-input dataset must be private and owned by the user. Both recovery GPU jobs are complete and released their GPUs. They consumed 2.11h of Kaggle quota in total, within the 3h recovery envelope; 25.37h remained afterward (reset 2026-11-14). Monitor with `kaggle kernels status` and `kaggle kernels logs --follow`; an observer connection failure alone does not mean the job failed. Do not reapply an edit to already edited weights.

Small evidence is saved under `/kaggle/working/qwen35_corrected/results`; the merged checkpoint uses `/kaggle/temp` before private Hub upload. Keep `SAVE_MERGED=True` for the complete save/reload/persistence workflow. Root-level notebooks are self-contained; uploading this folder is unnecessary.
