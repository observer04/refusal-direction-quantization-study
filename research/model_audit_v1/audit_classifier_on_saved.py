"""Run saved public text-only classifiers on already reviewed outputs; no models."""
import ast,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
source=(H/'reference/classifier_initial.py').read_text();tree=ast.parse(source)
# Execute definitions only, excluding project imports and CLI entrypoint.
allowed=[]
for n in tree.body:
 if isinstance(n,ast.ImportFrom) and n.module in ['__future__','src']:continue
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign,ast.AnnAssign)):allowed.append(n)
ns={'__name__':'audit_text_classifier','RESULTS_VERSION':'public_source_snapshot'};exec(compile(ast.Module(body=allowed,type_ignores=[]),'<saved_public_classifier_v3>','exec'),ns)
assert ns['VERSION']=='3.0'
metric=json.loads((H/'historical_metric_audit.json').read_text());markers=metric['markers']
def marker(text):
 if not text.strip():return True
 t=' '.join(text.lower().replace('*','').replace('’',"'").split());return any(m.lower() in t for m in markers)
queue={r['blind_id']:r for r in map(json.loads,(H/'saved_blind_queue.jsonl').read_text().splitlines())};labs={r['blind_id']:r for r in map(json.loads,(H/'saved_blind_labels.jsonl').read_text().splitlines())};mapping=list(map(json.loads,(H/'saved_blind_mapping.jsonl').read_text().splitlines()));rows=[]
for m in mapping:
 q=queue[m['blind_id']];lab=labs[m['blind_id']]
 if q['kind']!='safety':continue
 isref,layer,reason,conf=ns['classify_item']({'response':q['response'],'reasoning':'','behavior_id':m['id']})
 rows.append({'blind_id':m['blind_id'],'id':m['id'],'condition':m['condition'],'upstream_marker_refusal':marker(q['response']),'public_v3_is_refusal_without_overrides':isref,'public_v3_reason':reason,'public_v3_confidence':conf,'independent_requested_fulfillment':lab['requested_fulfillment'],'independent_safety_withholding':lab['safety_withholding'],'independent_fact_correction':lab['factual_correction'],'length_cap':q['length_cap']})
(H/'classifier_comparison_saved.json').write_text(json.dumps({'scope':'Public v3 source on OUR saved nonthinking outputs, no external manual overrides. NOT reproduction of external400 results. Independent labels frozen first.','v3_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
