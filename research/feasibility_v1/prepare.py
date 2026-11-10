"""Stdlib-only artifact preparation; never loads a model locally."""
import ast
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def sha(blob): return hashlib.sha256(blob).hexdigest()
def key(s): return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC',s).casefold()).split())
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

exposed=[]
for p in [ROOT/'reproduction/qwen35_9b/recovery/runs/pilot-v1/qwen35_recovery/results/prompts.jsonl',
          ROOT/'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/prompts.jsonl',
          ROOT/'reproduction/qwen35_9b/recovery/paired_long_prompts.json',
          ROOT/'reproduction/qwen35_9b/recovery/benign_smoke.jsonl']:
    rows=json.loads(p.read_text()) if p.suffix=='.json' else loadl(p)
    exposed.extend(dict(prompt=r['prompt'],file=str(p.relative_to(ROOT)),split=r.get('split','long_or_smoke')) for r in rows)
used={key(r['prompt']) for r in exposed}
hb=list(csv.DictReader((HERE/'reference/harmbench_test.csv').open()))
# Frozen category strata; exclude chemical/biological and copyrighted tasks.
strata=['cybercrime_intrusion','illegal','misinformation_disinformation','harmful']
harm=[]
for category in strata:
    eligible=[r for r in hb if r['FunctionalCategory']=='standard' and not r['ContextString'].strip()
              and r['SemanticCategory']==category and key(r['Behavior']) not in used]
    eligible.sort(key=lambda r:sha(('feasibility-v1-20261110:'+r['BehaviorID']).encode()))
    assert len(eligible)>=2
    for r in eligible[:2]:
        harm.append(dict(id='HarmBench:test:'+r['BehaviorID'],prompt=r['Behavior'],kind='safety',category=category,
                         source='centerforaisafety/HarmBench',max_new_tokens=768))
xs=list(csv.DictReader((HERE/'reference/xstest.csv').open()))
safe=[]
for category in ['homonyms','figurative_language','safe_targets','safe_contexts']:
    eligible=[r for r in xs if r['label']=='safe' and r['type']==category and key(r['prompt']) not in used]
    if not eligible:
        continue
    eligible.sort(key=lambda r:sha(('feasibility-v1-20261110:'+r['id']).encode()))
    r=eligible[0]
    safe.append(dict(id='XSTest:'+r['id'],prompt=r['prompt'],kind='sensitive_benign',category=category,
                     source='paul-rottger/xstest',max_new_tokens=768))
assert len(safe)==4, [(r['type'],r['label']) for r in xs[:5]]
gsm=loadl(HERE/'reference/gsm8k_test.jsonl')
eligible=[(i,r) for i,r in enumerate(gsm) if len(r['question'])<=280 and key(r['question']) not in used]
eligible.sort(key=lambda pair:sha(('feasibility-v1-20261110:gsm:'+str(pair[0])).encode()))
utility=[]
for i,r in eligible[:4]:
    utility.append(dict(id='GSM8K:test:'+str(i),prompt=r['question'],kind='utility',category='short_arithmetic',
                        source='openai/grade-school-math',expected_answer=r['answer'].split('####')[-1].strip(),
                        reference_answer=r['answer'],max_new_tokens=384))
records=[]
for f in range(4):
    for r in [harm[2*f],harm[2*f+1],safe[f],utility[f]]:
        r=dict(r,family=f+1,prompt_sha256=sha(r['prompt'].encode()),normalized_sha256=sha(key(r['prompt']).encode()))
        assert key(r['prompt']) not in used
        used.add(key(r['prompt']));records.append(r)
(HERE/'prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
adapter=ROOT/'reproduction/qwen35_9b/recovery/runs/corrected-v1/qwen35_corrected/results/adapter'
pins={'original':{'id':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
      'corrected':{'id':'Observer04/Qwen3.5-9B-heretic-corrected-20261109','revision':'6cd66cf9bf38efab01ffd5ab6e5c7995868f55bb'},
      'adapter_remote':'recovery_evidence/corrected-v1/adapter',
      'adapter_sha256':{p.name:sha(p.read_bytes()) for p in adapter.iterdir() if p.name in ['adapter_config.json','adapter_model.safetensors']},
      'heretic_commit':'e26da5e0e648bb285f896c9801849edfa8d20913',
      'source_manifest':json.loads((HERE/'reference/source_manifest.json').read_text())}
(HERE/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
report={'exposed_records_checked':len(exposed),'exposed_unique_normalized':len({key(r['prompt']) for r in exposed}),
        'exposed_splits':{s:sum(r['split']==s for r in exposed) for s in sorted({r['split'] for r in exposed})},
        'new_prompt_count':len(records),'new_exact_or_normalized_overlaps':0,
        'dedup_scope':'NFKC, casefold, punctuation and whitespace normalized equality against full saved pilot extraction/historical/development/fresh fixtures, corrected extraction fixtures, twelve long diagnostics and benign smoke. Semantic near-duplicates and pretraining/benchmark contamination unresolved.',
        'prompts_sha256':sha((HERE/'prompts.jsonl').read_bytes())}
(HERE/'dedup_verification.json').write_text(json.dumps(report,indent=2)+'\n')
for p in HERE.glob('*.py'): ast.parse(p.read_text())
sources={n:(HERE/n).read_text() for n in ['PROTOCOL.md','review_rubric.md','prepare.py','run_kaggle.py','prompts.jsonl','pins.json','dedup_verification.json']}
rendered=(HERE/'run_kaggle.py').read_text().replace('__PINS__',repr(pins)).replace('__PROMPTS__',repr(records)).replace('__SOURCE_FILES__',repr(sources))
ast.parse(rendered)
(HERE/'kaggle/feasibility.py').write_text(rendered)
write_manifest={str(p.relative_to(HERE)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in HERE.rglob('*')
                if p.is_file() and p.suffix in ['.py','.json','.jsonl','.md','.csv'] and '__pycache__' not in p.parts
                and p.name not in ['preregistration_manifest.json']}
(HERE/'preregistration_manifest.json').write_text(json.dumps(write_manifest,indent=2)+'\n')
print(json.dumps(report,indent=2))
