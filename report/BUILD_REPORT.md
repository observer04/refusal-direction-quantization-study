# Report source and build

The editable deliverable is `Refusal_Direction_Ablation_Report.docx`. It follows the retained `../SampleReport.docx` college-report format. The sample is never modified.

`report_content.py` holds the prose, tables, bibliography and planned page grouping. `build_report.py` patches the retained DOCX package and creates the plain workflow diagram. Its `equation_xml` function holds the four native Word equations; the equation strings in the content source are readable source descriptions. No local model is loaded and no experiment is rerun by this build.

Build from the project root with the bundled document runtime:

```bash
/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 report/build_report.py
```

The build verifies the sample SHA-256 and preserves styles, footer parts, theme, numbering and other untouched package parts. It requires the bundled `lxml`, Pillow and the documents skill's OOXML helper. No installation or virtual environment is needed.

Render and inspect after changing content or metadata:

```bash
/home/op/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 /home/op/.codex/plugins/cache/openai-primary-runtime/documents/26.1105.12137/skills/documents/render_docx.py report/Refusal_Direction_Ablation_Report.docx --output_dir report/writing_qa/render-next --emit_pdf
```

Use the bundled LibreOffice through this renderer. The QA PDF and PNGs are internal artifacts. The DOCX is the requested deliverable.

Contents and list pages contain bookmark links with static page locators. Their current values match the verified 26-page revised render. If editing changes pagination, update the page grouping/locators in the build and verify again. Retained footer PAGE fields are set to refresh on opening in Word. Word's exact pagination can differ with fonts and printer settings, so check the submission copy after inserting names, a logo or additional text.

Before submission, replace the clearly marked student, ID, supervisor, programme, department, institute and emblem placeholders. Personal acknowledgements are optional editable text. Examination decisions, dates and signatures remain for the authorized examiners to complete.

Author-only claim-to-source mapping is in `CLAIM_TRACEABILITY.md`; it is intentionally separate from the self-contained report. Compact QA records are under `writing_qa/`; generated page images and PDF renders are excluded from Git. The records document the 26-page revised inspection, package preservation, source-template contract and bibliography ordering. The report describes a focused literature review, audited reproduction and limited feasibility pilot; it does not claim a new algorithm, verified harmful effectiveness, complete benchmark coverage or a resolved quantization mechanism.
