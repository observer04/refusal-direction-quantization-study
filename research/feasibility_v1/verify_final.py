"""Audit completed feasibility artifacts without loading a model or dependencies."""
import ast
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/run-v1/output/feasibility_v1/results'

def readl(name):
    return [json.loads(line) for line in (OUT / name).read_text().splitlines() if line]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class ReviewHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.articles = []
        self.current = None
        self.pre = None
        self.textareas = 0
        self.external_resources = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article':
            self.current = []
            self.articles.append(self.current)
        if tag == 'pre' and self.current is not None:
            self.pre = []
        if tag == 'textarea':
            self.textareas += 1
        if tag in ['script', 'img', 'iframe']:
            self.external_resources.append((tag, attrs))
    def handle_data(self, data):
        if self.pre is not None:
            self.pre.append(data)
    def handle_endtag(self, tag):
        if tag == 'pre' and self.pre is not None:
            self.current.append(''.join(self.pre))
            self.pre = None
        if tag == 'article':
            self.current = None

def main():
    prereg = json.loads((HERE / 'preregistration_manifest.json').read_text())
    for name, entry in prereg.items():
        assert digest(HERE / name) == entry['sha256'], ('Preregistered file changed', name)
    rows = readl('all_responses.jsonl')
    assert len(rows) == 60
    assert Counter(r['condition'] for r in rows) == {c: 12 for c in ['W', 'D', 'QW', 'QD', 'QW_D']}
    per_arm = []
    for c in ['D', 'QD', 'QW_D', 'QW', 'W']:
        per_arm.extend(readl(c + '_responses.jsonl'))
    assert rows == per_arm, 'Combined raw output differs from saved per-arm records'
    queue = readl('blind_review_queue.jsonl')
    mapping = readl('blind_review_mapping.jsonl')
    labels = readl('blind_agent_labels.jsonl')
    assert len(queue) == len(mapping) == len(labels) == 60
    assert len({r['review_id'] for r in queue}) == 60
    assert {r['review_id'] for r in queue} == {r['review_id'] for r in labels}
    by_key = {(r['condition'], r['id']): r for r in rows}
    by_review = {r['review_id']: r for r in mapping}
    for q in queue:
        m = by_review[q['review_id']]
        r = by_key[(m['condition'], m['prompt_id'])]
        assert q['prompt'] == r['prompt'] and q['response'] == r['response']
        assert q['raw_completion'] == r['raw_completion']
        assert m['response_sha256'] == hashlib.sha256(r['response'].encode()).hexdigest()
        assert not {'condition', 'model', 'id', 'expected_answer'} & q.keys()
        assert all(v is None or v == '' for v in q['human_ratings'].values())
    frozen = json.loads((HERE / 'verification/labels_frozen_before_unmask.json').read_text())
    assert digest(OUT / 'blind_agent_labels.jsonl') == frozen['labels_sha256']
    assert digest(HERE / 'analyze.py') == frozen['analysis_sha256']
    selected = json.loads((OUT / 'selected_prompts.json').read_text())
    order = {r['id']: i for i, r in enumerate(selected)}
    arms = {c: i for i, c in enumerate(['W', 'D', 'QW', 'QW_D', 'QD'])}
    paired = sorted(rows, key=lambda r: (order[r['id']], arms[r['condition']]))
    html_results = {}
    for filename, source, keys in [
        ('HUMAN_REVIEW_ALL.html', paired, ['prompt', 'response', 'raw_completion', 'raw_output']),
        ('HUMAN_REVIEW_BLINDED.html', queue, ['prompt', 'response', 'raw_completion']),
    ]:
        parser = ReviewHTML()
        parser.feed((OUT / filename).read_text())
        assert len(parser.articles) == parser.textareas == 60
        assert not parser.external_resources
        assert parser.articles == [[r[k] for k in keys] for r in source], filename
        html_results[filename] = {'cases': 60, 'exact_prompt_and_raw_completion_text': True,
                                  'blank_rating_widgets': 60, 'embedded_active_resources': 0}
    assert not readl('interrupted_token_records.jsonl')
    verification = json.loads((OUT / 'artifact_verification.json').read_text())
    assert verification['verified'] and verification['unfinalized_journal_records'] == 0
    merge = json.loads((OUT / 'frozen_delta_merge_verification.json').read_text())
    adapter = json.loads((OUT / 'QW_D_adapter_verification.json').read_text())
    native = json.loads((OUT / 'native_quantization_verification.json').read_text())
    assert merge['all_writer_hashes_match'] and len(merge['checks']) == 64
    assert adapter['all64_factors_exact'] and adapter['same_native_packed_state']
    assert len(adapter['module_checks']) == 64
    assert all(x['max_abs_error'] <= x['tolerance'] for x in adapter['module_checks'])
    assert native['native_linear4bit_modules'] == 358
    assert len(native['changed_modules']) == len(native['changed_absmax_grids']) == 42
    assert native['no_changes_outside_active_writers']
    assert native['QW_D_exact_same_packed_backbone'] and native['refitted_scales_not_fixed_grid']
    for path in [HERE / n for n in ['prepare.py', 'run_kaggle.py', 'kaggle/feasibility.py',
                                    'build_review.py', 'analyze.py', 'persist.py', 'verify_final.py']]:
        ast.parse(path.read_text(), filename=str(path))
    report = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'verified': True,
              'preregistered_files_unchanged': len(prereg), 'all_completions_preserved': 60,
              'raw_combined_equals_per_arm_records': True, 'human_fields_blank_and_agent_labels_separate': True,
              'condition_mapping_separate': True, 'frozen_blinded_labels_and_analysis_unchanged': True,
              'html_roundtrip_text_checks': html_results, 'no_unfinalized_generations': True,
              'merge_writer_hashes_pass': 64, 'adapter_module_checks_pass': 64,
              'native_packed_modules': 358, 'changed_active_writers_and_scales': 42,
              'scope': 'Local stdlib artifact verification. No new inference or independent semantic reviewer.'}
    (HERE / 'verification/final_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
