"""Stdlib-only frozen artifact preparation; model work is Kaggle-only."""
import ast,csv,hashlib,json,re,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def key(s):return ' '.join(re.sub(r'[^\w\s]',' ',unicodedata.normalize('NFKC',s).casefold()).split())
def loadl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
base=ROOT/'research/feasibility_v1'; old=base/'runs/run-v1/output/feasibility_v1/results'
matched=json.loads((old/'selected_prompts.json').read_text());assert len(matched)==12
exposed=[]
for p in [ROOT/'reproduction/qwen35_9b/recovery/runs/pilot-v1/qwen35_recovery/results/prompts.jsonl',ROOT/'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/prompts.jsonl',ROOT/'reproduction/qwen35_9b/recovery/paired_long_prompts.json',ROOT/'reproduction/qwen35_9b/recovery/benign_smoke.jsonl',base/'prompts.jsonl']:
 exposed.extend(json.loads(p.read_text()) if p.suffix=='.json' else loadl(p))
used={key(r['prompt']) for r in exposed}
hb=list(csv.DictReader((base/'reference/harmbench_test.csv').open()));xs=list(csv.DictReader((base/'reference/xstest.csv').open()));gsm=loadl(base/'reference/gsm8k_test.jsonl')
fresh=[]
for f,category in enumerate(['cybercrime_intrusion','illegal','misinformation_disinformation','harassment_bullying'],1):
 elig=[r for r in hb if r['FunctionalCategory']=='standard' and not r['ContextString'].strip() and r['SemanticCategory']==category and key(r['Behavior']) not in used]
 # Contact lookup is a knowledge/retrieval test; exclude before selection, not after responses.
 elig=[r for r in elig if not any(t in r['Behavior'].lower() for t in ['contact','vendors','where to buy','where can i buy','locating','supplier','pipeline','arson','suicide','self-harm','child','consent','antivirus','utility company','explosive','bomb','weapon','terror','kill','murder','poison','fentanyl','cyanide','ricin','nuclear','traffick','holocaust','hostage','death threat','cdc','explicit','intimate','lgbtq','immigration'])]
 elig.sort(key=lambda r:sha(('model-audit-v1-20261110:'+r['BehaviorID']).encode()));assert len(elig)>=4
 for r in elig[:4]:
  row=dict(id='HarmBench:test:'+r['BehaviorID'],prompt=r['Behavior'],kind='safety',category=category,source='centerforaisafety/HarmBench',max_new_tokens=1024,family=f)
  fresh.append(row);used.add(key(row['prompt']))
 category_x=['homonyms','figurative_language','safe_targets','safe_contexts'][f-1]
 eligx=[r for r in xs if r['label']=='safe' and r['type']==category_x and key(r['prompt']) not in used];eligx.sort(key=lambda r:sha(('model-audit-v1-20261110:'+r['id']).encode()));r=eligx[0]
 row=dict(id='XSTest:'+r['id'],prompt=r['prompt'],kind='sensitive_benign',category=category_x,source='paul-rottger/xstest',max_new_tokens=768,family=f);fresh.append(row);used.add(key(row['prompt']))
 eligg=[(i,r) for i,r in enumerate(gsm) if len(r['question'])<=280 and key(r['question']) not in used];eligg.sort(key=lambda x:sha(('model-audit-v1-20261110:gsm:'+str(x[0])).encode()));i,r=eligg[0]
 row=dict(id='GSM8K:test:'+str(i),prompt=r['question'],kind='utility',category='short_arithmetic',source='openai/grade-school-math',expected_answer=r['answer'].split('####')[-1].strip(),reference_answer=r['answer'],max_new_tokens=384,family=f);fresh.append(row);used.add(key(row['prompt']))
groups=[fresh[i:i+6] for i in range(0,24,6)]
fresh=[dict(r,family=i+1) for i in range(4) for r in [*[groups[c][i] for c in range(4)],*groups[i][4:]]]
for r in fresh:r.update(prompt_sha256=sha(r['prompt'].encode()),normalized_sha256=sha(key(r['prompt']).encode()))
assert len(fresh)==24 and not ({key(r['prompt']) for r in fresh}&{key(r['prompt']) for r in exposed})
pins=json.loads((base/'pins.json').read_text());pins['publisher']={'id':'trohrbaugh/Qwen3.5-9B-heretic-v2','revision':'81dda62421c43a5bd5722edbc6cd47714673dc7f'}
(HERE/'pins.json').write_text(json.dumps(pins,indent=2)+'\n');(HERE/'matched_prompts.json').write_text(json.dumps(matched,indent=2)+'\n');(HERE/'fresh_prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fresh))
(HERE/'dedup_verification.json').write_text(json.dumps({'exposed_records':len(exposed),'exposed_unique_normalized':len({key(r['prompt']) for r in exposed}),'fresh_count':24,'normalized_overlaps':0,'limitations':'Exact/normalized equality only; semantic and pretraining overlap unresolved.'},indent=2)+'\n')
if (HERE/'run_kaggle.py').exists():
 sources={n:(HERE/n).read_text() for n in ['PROTOCOL.md','review_rubric.md','prepare.py','run_kaggle.py','pins.json','matched_prompts.json','fresh_prompts.jsonl','dedup_verification.json','matched_reuse_metadata.json']}
 rendered=(HERE/'run_kaggle.py').read_text().replace('__PINS__',repr(pins)).replace('__MATCHED__',repr(matched)).replace('__FRESH__',repr(fresh)).replace('__SOURCE_FILES__',repr(sources));ast.parse(rendered);(HERE/'kaggle/audit.py').write_text(rendered)
manifest={str(p.relative_to(HERE)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.jsonl'] and 'runs' not in p.parts and p.name!='preregistration_manifest.json'}
(HERE/'preregistration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps([{'family':r['family'],'id':r['id'],'prompt':r['prompt']} for r in fresh],indent=2))
