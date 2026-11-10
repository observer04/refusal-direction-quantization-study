# Refusal Direction Ablation and Quantization Stability

A focused literature review and exploratory Qwen3.5-9B reproduction study, with a native NF4 quantization pilot. This repository preserves the college report, its editable sources, Kaggle experiment code and compact research records.

## Read the report

- [Revised college report](report/Refusal_Direction_Ablation_Report.docx): 26 pages, using the supplied college-report layout and IEEE-style numbered citations.
- [Frozen twelve-source bibliography](report/BIBLIOGRAPHY_PLAN.md).
- [Report revision and source-grounding record](report/TRIAGE_AND_FIX.md).
- [Current project state](STATE.md).

Student, institution, examination and signature fields remain editable placeholders. Check pagination after filling them. The report is a scoped empirical study, with no claim of a new ablation method or a validated quantization mechanism.

## What the evidence establishes

The corrected reconstruction recovers a missing upstream harmless-mean projection step. Under the historical harness, the original model has 100/100 refusal-marker matches and the publisher and corrected reconstruction each have 8/100. The corrected checkpoint produces 100/100 identical responses after save/reload. The historical test was exposed during diagnosis, and 99/100 corrected responses reach its short token cap. Marker counts are not harmful-compliance rates.

A separate semantic audit finds matching fulfillment categories for publisher and reconstruction on fourteen selected safety cases. On eight fresh cases, each edited model has six clear-fulfillment judgments, compared with one for the original. These are single-agent judgments with imperfect masking, unverified technical correctness and substantial truncation; they are not population estimates or a test of broad model equivalence.

The five-condition NF4 pilot establishes implementation feasibility. Its small fixture does not resolve whether quantization changes the effect of the fixed edit. Refitted scales and a separate frozen floating edit are treated explicitly; no direction is re-extracted from a quantized model. Five selected safe XSTest items and elementary arithmetic provide narrow checks, not complete benchmark scores.

## Experiment entry points

All model loading, package installation, direction extraction and inference belong on **Kaggle**, not a local virtual environment.

1. Upload [the corrected reconstruction notebook](Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb) to a fresh Kaggle session with Internet and the tested T4 ×2 configuration.
2. Supply any required Hugging Face credential privately through Kaggle Secrets or the runtime configuration. Do not put it in notebook source or outputs.
3. Read the [reproduction guide](reproduction/qwen35_9b/README.md) and pinned settings before running. Historical failed source notebooks remain for context.

The bounded semantic and NF4 jobs are complete. Rerunning inference consumes GPU quota and is not required to read this repository or rebuild the report.

The historical Heretic release is an upstream Git submodule. Use `git submodule update --init` only if you need that older reference source; the corrected Kaggle notebook independently fetches its pinned compatible source.

## Repository guide

| Location | Contents |
|---|---|
| `report/` | Final DOCX, content/build source, bibliography, revision and traceability notes |
| `SampleReport.docx` | Unchanged layout reference required by the report builder |
| `reproduction/qwen35_9b/` | Pinned reproduction materials, recovery code and historical interpretation |
| `research/model_audit_v1/` | Final implementation/semantic/metric audit and its source |
| `research/feasibility_v1/` | Frozen native NF4 protocol, analysis and findings |
| `research/round2/` and `research/recommendation_*.md` | Research planning and adversarial review, not measured findings |
| `scripts/` | Kaggle and persistence helpers |

[AUDIT.md](research/model_audit_v1/AUDIT.md) and [NF4 RESULTS.md](research/feasibility_v1/RESULTS.md) are the authoritative result summaries. Some older notes describe earlier project states; their dates and scope matter.

## Rebuild the document

See [BUILD_REPORT.md](report/BUILD_REPORT.md). The current builder uses the existing bundled document runtime and its OOXML helper. It is not a standalone dependency installer. Rebuilding the DOCX does not load a language model. Render and inspect the resulting document after any layout or metadata change.

## Evidence persistence

Full model weights, adapter tensors, raw completions, token journals, output-bearing notebooks and rendered QA images are excluded from Git. The measured private checkpoint and evidence bundles are already persisted separately on Hugging Face under the project owner's account; immutable revisions and file hashes are recorded in the research notes. GitHub access does not grant access to those private bundles. Some detailed companion links therefore require the retained workspace or separately authorized evidence access.

No credentials are included. Upstream software and benchmark material remain attributed in the report and [third-party source notes](THIRD_PARTY_SOURCES.md).
