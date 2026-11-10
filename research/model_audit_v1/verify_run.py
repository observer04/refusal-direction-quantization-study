"""Stdlib-only verification of saved audit records; no model/tokenizer loading."""
import collections,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
OUT=H/'runs/run-v1/output/model_audit_v1/results'
def load(p):return json.loads(p.read_text())
def loadl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 frozen=load(H/'preregistration_manifest.json')
 mismatches=[n for n,d in frozen.items() if sha(H/n)!=d['sha256']]
 assert mismatches==['reference/frozen_fixture_listing.json'],mismatches
 auxiliary=load(H/'verification/auxiliary_listing_hash_discrepancy.json')
 assert auxiliary['listing_matches_exact_projection_of_frozen_fresh_fixture'] and sha(H/mismatches[0])==auxiliary['actual_sha256']
 manifest=load(OUT/'run_manifest.json');rows=[]
 for c in ['P','W','D']:
  p=OUT/(c+'_responses.jsonl')
  if p.exists():rows.extend(loadl(p))
 assert len(rows)<=84 and len(rows)==manifest['completed_responses']
 assert len({(r['condition'],r['id']) for r in rows})==len(rows)
 events=loadl(OUT/'generation_token_journal.jsonl');starts={};tokens=collections.defaultdict(list)
 for e in events:
  key=e['condition'],e['id']
  if e['event']=='start':assert key not in starts;starts[key]=e
  elif e['event']=='tokens':tokens[key].extend(e['token_ids'])
  else:raise AssertionError(e['event'])
 for r in rows:
  k=r['condition'],r['id'];s=starts[k]
  assert tokens[k]==r['generated_token_ids'] and len(tokens[k])==r['generated_tokens']
  assert len(s['input_ids'])==r['input_tokens']
  assert hashlib.sha256(s['rendered_prompt'].encode()).hexdigest()==r['input_rendered_sha256']
  assert s['rendered_prompt']==r['rendered_prompt'] and s['prompt']==r['prompt']
  assert s['max_new_tokens']==r['max_new_tokens']==r['generation_settings']['max_new_tokens']
  assert r['generated_tokens']<=r['max_new_tokens']
  assert r['hit_token_limit']==(not r['eos_seen'] and r['generated_tokens']>=r['max_new_tokens'])
  eos=r['generation_settings']['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
  indexes=[i for i,t in enumerate(tokens[k]) if t in eos]
  assert (bool(indexes)==r['eos_seen']) and (not indexes or indexes==[len(tokens[k])-1])
  assert r['generation_settings']['do_sample'] is False and r['generation_settings']['enable_thinking'] is False
  assert r['generation_settings']['batch_size']==1 and r['seed']==0
  assert r['model']==manifest['pins'][{'W':'original','D':'corrected','P':'publisher'}[r['condition']]]
 grouped=collections.defaultdict(list)
 for r in rows:grouped[r['id']].append(r)
 for rs in grouped.values():
  assert len({r['input_rendered_sha256'] for r in rs})==1
  assert len({json.dumps(r['generation_settings'],sort_keys=True) for r in rs})==1
  assert len({r['prompt'] for r in rs})==1
 reuse=load(OUT/'matched_reuse_verification.json');assert reuse['all'] and len(reuse['records'])==12
 profile=load(OUT/'profile_decision.json') if (OUT/'profile_decision.json').exists() else None
 selected=load(OUT/'selected_fresh_prompts.json') if (OUT/'selected_fresh_prompts.json').exists() else []
 if manifest['status']=='all_selected_conditions_complete':
  assert len(selected)==6*profile['selected_family_prefix']
  assert len(rows)==12+3*len(selected)
  assert all({r['condition'] for r in grouped[q['id']]}=={'P','W','D'} for q in selected)
 counts=collections.Counter((r['condition'],r['cohort'],r['kind']) for r in rows)
 report={'manifest_status':manifest['status'],'new_completion_count':len(rows),'cap_84_observed':len(rows)<=84,'counts':{' | '.join(k):v for k,v in sorted(counts.items())},'journal_start_count':len(starts),'journal_token_sequence_count':len(tokens),'journal_matches_all_finalized_rows':True,'unfinalized_starts':[list(k) for k in starts if k not in {(r['condition'],r['id']) for r in rows}],'profile':profile,'selected_fresh_prompts':len(selected),'matched_reuse_gate_passed':True,'elapsed_seconds':manifest['elapsed_seconds'],'failures':manifest['failures'],'all_substantive_preregistration_hashes_unchanged':True,'all_preregistration_hashes_unchanged':False,'auxiliary_listing_hash_exception':auxiliary,'file_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},'limits':'Saved records and in-job assertions audited; no local weights or model/tokenizer execution. This verifier does not independently recompute logits or decoded token text.'}
 (H/'verification/run_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['manifest_status','new_completion_count','selected_fresh_prompts','elapsed_seconds','failures','unfinalized_starts']},indent=2))
if __name__=='__main__':main()
