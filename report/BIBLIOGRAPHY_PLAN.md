# Frozen report bibliography

Updated 10 November 2026. The user requested two additional recent quantization/refusal papers after the proposed ten-entry list: the following twelve source identities are frozen for the report and viva preparation. Source replacement or expansion requires an invalid-source finding; none was found in the triage. Table numbers below are planning identifiers, not final citation numbers. This is a source plan, not the finished report bibliography or a claim that all studies were reproduced. No new inference is authorized by this update.

| No. | Source | Purpose |
|---|---|---|
| 1 | Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) | Central method and intervention background. |
| 2 | Souly et al. (2024), [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) | Evaluation validity; refusal wording versus useful fulfillment. Our rubric is informed by this work; its full evaluator was not run. |
| 3 | Mazeika et al. (2024), [HarmBench](https://arxiv.org/abs/2402.04249) | Selected safety prompt provenance and evaluation framework. We did not reproduce its full official evaluation. |
| 4 | Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer foundations; distinguish Qwen's hybrid architecture. |
| 5 | Dettmers et al. (2023), [QLoRA](https://arxiv.org/abs/2305.14314) | NF4 background. No QLoRA training, double quantization or paged optimizer was used. |
| 6 | Chen et al. (2025), [Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](https://proceedings.mlr.press/v267/chen25ci.html) | Quantization-safety prior work, including Q-resafe. |
| 7 | Röttger et al. (NAACL 2024; preprint 2023), [XSTest](https://arxiv.org/abs/2308.01263) | Actual selected sensitive-benign fixtures; over-refusal. |
| 8 | Qwen team, [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) | Architecture/model provenance; record immutable revision in methods. |
| 9 | p-e-w, [Heretic software at the experiment's pinned revision](https://github.com/p-e-w/heretic/tree/e26da5e0e648bb285f896c9801849edfa8d20913) | Recovered implementation recipe; fixed recipe replay rather than our own optimizer search. |
| 10 | Abliterlitics, [Qwen3.5-9B comparison](https://abliterlitics.dev/models/qwen3.5-9b/) and [audited classifier snapshot](https://github.com/dreamfast/abliterlitics/blob/c48b882660daffd06f1c01ba59f1e3d114f768f8/src/benchmark/harmbench_classify.py) | External claim and source of the audited heuristic; one project reference with report/code locators. Historical 0/400 remains unreplicated. |
| 11 | Sahil Kadadekar (8 June 2026), [Quality Is Not a Safety Proxy Under Quantization](https://arxiv.org/abs/2606.10154) | Recent quantization/evaluation study. **Preprint**, not a verified conference publication. |
| 12 | Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani and Husrev Taha Sencar (EMNLP 2026), [There Is More to Refusal in Large Language Models than a Single Direction](https://aclanthology.org/2026.emnlp-main.2606/) | Recent qualification of the single-direction interpretation. |

## Why the additions fit

Kadadekar reports that retained quality can coexist with reduced refusal on the measured model/quantizer matrix. Refusal-direction probes were weak separators of problematic conditions; the proposed behavioral screen is calibrated within that matrix, with external validation unresolved. This motivates separate safety/utility testing and prospective validation of any future edit-stability predictor. It does not test our reconstructed edit or establish an NF4 effect here. Public code/data access is incomplete in the inspected manuscript.

Joad et al. report category-dependent activation directions alongside broadly shared steering trade-offs, and analyze shared versus category-specific latent features. This supports caution when interpreting one successful intervention as a complete account of refusal. Our study has not performed their sparse-autoencoder or category-geometry analysis, and the paper is not evidence that our specific redirects have the same mechanism.

## XSTest actually used

Verified from executed selected fixtures, not only the proposed protocols:

| Stage | Executed safe items | Arms |
|---|---|---|
| Native NF4 feasibility | `XSTest:24` download throttling; `XSTest:63` chess; `XSTest:104` fish preparation | Five conditions, three items each. |
| Independent audit, matched | Same three items | Original, publisher and reconstruction; original/reconstruction records reused after settings checks. |
| Independent audit, fresh | `XSTest:6` photography; `XSTest:68` opening a beer | Original, publisher and reconstruction, two new items each. |

**Five distinct safe XSTest prompts**, not a full benchmark run. No safety over-refusal was observed on these cases under the recorded agent review. Some answers were partially useful or capped, so this does not establish general over-refusal performance or broad utility preservation.

Evidence: [feasibility results](../research/feasibility_v1/RESULTS.md), [executed pilot fixture](../research/feasibility_v1/runs/run-v1/output/feasibility_v1/results/selected_prompts.json), [executed fresh audit fixture](../research/model_audit_v1/runs/run-v1/output/model_audit_v1/results/selected_fresh_prompts.json), [audit](../research/model_audit_v1/AUDIT.md).

Keep all exact checkpoint, code and dataset origins/revisions in the provenance appendix, including the selected GSM8K arithmetic fixtures. Do not describe small subsets as complete benchmark scores. Final citation style and numbering should follow the finished report's order of first citation.
