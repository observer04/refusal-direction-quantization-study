"""Stdlib-only audit of original substring scorer and saved historical records."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];S=R/'reproduction/qwen35_9b/heretic-qwen35/src/heretic'
module=ast.parse((S/'config.py').read_text());markers=None
for node in ast.walk(module):
 if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='refusal_markers':
  markers=ast.literal_eval(next(k.value for k in node.value.keywords if k.arg=='default'))
assert len(markers)==32
em=ast.parse((S/'evaluator.py').read_text());fn=next(n for n in ast.walk(em) if isinstance(n,ast.FunctionDef) and n.name=='is_refusal')
# Run exact extracted stdlib-only method with a tiny settings object, never import model/torch.
ns={};exec(ast.unparse(fn),ns)
class Obj:pass
obj=Obj();obj.settings=Obj();obj.settings.refusal_markers=markers
refuse=lambda s:ns['is_refusal'](obj,s)
paths={
 'original':'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/baseline_responses.jsonl',
 'failed_raw':'reproduction/qwen35_9b/runs/batch-v1/qwen35_reproduction/results/edited_responses.jsonl',
 'publisher':'reproduction/qwen35_9b/runs/published-check-v1/qwen35_reproduction/results/published_fp16_responses.jsonl',
 'corrected':'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/edited_responses.jsonl',
 'reloaded':'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/reloaded_responses.jsonl'}
rows={};report={}
for name,rel in paths.items():
 p=R/rel;rr=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];rows[name]=rr
 labels=[refuse(r['response']) for r in rr]
 stored=[r['refusal_marker_match'] for r in rr]
 report[name]={'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':len(rr),'marker_refusals':sum(labels),'stored_label_field_present':all(x is not None for x in stored),'stored_labels_match':labels==stored,'cap_count':sum(r.get('hit_token_limit',False) for r in rr),'empty_count':sum(not r['response'].strip() for r in rr),'first_record_keys':sorted(rr[0])}
assert all(r['n']==100 for r in report.values())
assert all([r['prompt'] for r in rows[c]]==[r['prompt'] for r in rows['original']] for c in rows)
assert [r['response'] for r in rows['corrected']]==[r['response'] for r in rows['reloaded']]
(H/'historical_metric_audit.json').write_text(json.dumps({'source_sha256':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['config.py','evaluator.py','model.py']},'markers':markers,'normalization':'Empty -> True; lowercase, remove *, normalize curly apostrophe, collapse whitespace; any marker substring anywhere in decoded generated suffix.','conditions':report,'same_100_prompt_order':True,'corrected_reload_100_exact':True,'negative_marker_does_not_imply_fulfillment':True},indent=2)+'\n')
print(json.dumps(report,indent=2))
