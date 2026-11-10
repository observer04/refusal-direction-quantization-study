import hashlib,json,random
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'research/feasibility_v1/runs/run-v1/output/feasibility_v1/results'
rows=[]
for condition in ['W','D']:
 for s in (O/(condition+'_responses.jsonl')).read_text().splitlines():rows.append(json.loads(s))
random.Random(20261110517).shuffle(rows);queue=[];mapping=[]
for i,r in enumerate(rows,1):
 blind='S%03d'%i
 queue.append({'blind_id':blind,'prompt':r['prompt'],'response':r['response'],'kind':r['kind'],'generated_tokens':r['generated_tokens'],'length_cap':r['hit_token_limit'],'expected_answer':r.get('expected_answer'),'reference_answer':r.get('reference_answer'),'human_review':{'reviewer':None,'labels':None,'reason':None}})
 mapping.append({'blind_id':blind,'id':r['id'],'condition':r['condition'],'source_file':str(O/(r['condition']+'_responses.jsonl')),'source_row_sha256':hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()})
for n,rs in [('saved_blind_queue.jsonl',queue),('saved_blind_mapping.jsonl',mapping)]:
 p=H/n;assert not p.exists(),'Do not overwrite review';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
