# Final report QA — triage revision

10 November 2026. Final artifact: `report/Refusal_Direction_Ablation_Report.docx`.

SHA256: `149f6b2dc43653a5254d1f7b7d3112344a1b68ed252352c7bda5c00b562d24a8`.

The 28-page v1 was preserved before authoring at `report/baseline_v1_20261110/`. The revised report is 26 pages, with 9 front-matter pages, 14 chapter pages, 2 reference pages and 1 appendix page. It retains the supplied sample's single-column college typography and page system. The source sample is unchanged (SHA256 `e62bd2c6c049f854c84896c9ff903e4a04d583ca25ec51399f9895474e260f1a`).

## Content and citations

The abstract is 269 words and cites its two literature claims. Own numerical observations remain distinguished from external findings. Twelve bibliography source identities and their metadata exactly match v1 and are now frozen in the bibliography plan. All twelve are cited in first-appearance order, including abstract and table cells. Kadadekar is explicitly a preprint; Joad is identified as EMNLP 2026. No external source is made to validate our own semantic counts.

The revised literature review connects refusal direction selection, category-dependent representation, semantic evaluation, over-refusal and quantization safety. Table 2.1 synthesizes contributions and implications. Immediate RQ1–RQ3 are distinct from the prospective fixed-edit stability question. The report claims a focused review and exploratory baseline, with no new editing method, broad capability preservation, exact 0/400 replication, validated quantization failure or predictor.

Main results reproduce the authoritative historical, semantic audit and original NF4 pilot counts. The pilot labels and later audit remain separate. Threats include the small selected fixture, automated reviewer, imperfect masking, substantial truncation and unverified correctness. Operational hash/quota/packaging detail is absent from chapters. Exact workspace evidence remains in the separate `CLAIM_TRACEABILITY.md`.

## Rendering and visual review

Canonical `render_docx.py` using the bundled Python and bundled LibreOffice produced `triage-render-v1/`, including 26 page PNGs and an internal PDF. Every page was individually inspected at readable resolution. Cover and front matter, page numbers, contents and all lists, all 8 tables, the 1 figure, all 4 native displayed equations and both reference pages were checked. No overflowing content, clipped text, split table rows, orphan captions, broken image glyphs or blank pages were observed. Table headers, source IDs and reference URLs are readable.

Fresh `render_and_diff.py` output is in `triage-template-diff-final/`. The final `b_render/` PNGs match the separately inspected 26 pages byte-for-byte. Expected content differences against the 18-page sample are recorded there; representative source cover and chapter pattern were compared directly. Content expansion/compaction and removal of unconfirmed branding are intentional, documented in `artifact.md`.

## Package and self-containment checks

`build_manifest.json`, `structural_checks.json` and `self_containment_scan.json` describe the exact final artifact. There are 18 mapped sections, 8 data tables, 4 native numbered equations, 12 cited references and 26 page anchors. Contents/bookmark links all resolve and static page locators agree with actual rendered pages.

All 13 untouched original package parts remain identical and all original relationships remain. Only the five allowed existing parts change: document body, settings, relationships, content types and core metadata. One generated study diagram is added. Styles/theme/numbering/footers and the original image bytes remain unchanged. There are no comments, tracked changes or content controls.

Scanning all Word XML text/instructions, including paragraphs, cells, captions, headers, footers and bibliography, plus external relationship targets finds no internal research/reproduction route, private checkpoint name, unresolved citation token or substitute artifact reference. The report carries actual tables and explanations rather than pointing to internal JSON/notebooks.

## Remaining editable fields

Student name/ID, programme, supervisor, institute, department, emblem, examination date, decision/signatures and personal acknowledgement remain explicit placeholders. These cannot be supplied from verified context. No institutional approval or human ratings are asserted. No new model inference, GPU job, package installation or external upload occurred during revision.
