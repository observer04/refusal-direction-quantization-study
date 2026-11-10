"""Recheck raw boundaries, frozen labels, escaped human bundles and reproduction."""
import collections,hashlib,json,subprocess,sys
from html.parser import HTMLParser
from pathlib import Path
H=Path(__file__).resolve().parent
ll=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=ll(H/'all_responses.jsonl');assert len(rows)==72
 for prefix in ['saved','new']:
  f=json.loads((H/f'verification/{prefix}_labels_frozen_before_unmask.json').read_text())
  assert sha(H/f'{prefix}_blind_labels.jsonl')==f['labels_sha256']
  assert sha(H/f'{prefix}_blind_queue.jsonl')==f['queue_sha256']
 names=['all_responses.jsonl','human_blind_queue.jsonl','human_blind_mapping.jsonl','HUMAN_REVIEW_PAIRED.html','HUMAN_REVIEW_BLINDED.html','new_blind_queue.jsonl','new_blind_mapping.jsonl']
 before={n:sha(H/n) for n in names}
 subprocess.run([sys.executable,str(H/'build_review.py')],check=True)
 assert before=={n:sha(H/n) for n in names},'Review builder is not reproducible'
 queue=ll(H/'human_blind_queue.jsonl');assert len(queue)==72
 assert all(all(v is None for v in r['human_review'].values()) for r in queue)
 assert all('condition' not in r and 'id' not in r and 'model' not in r for r in queue)
 class Parse(HTMLParser):
  def __init__(self):super().__init__(convert_charrefs=True);self.pre=None;self.fields=[];self.prompts=[];self.responses=[];self.articles=0;self.textareas=0;self.intextarea=False;self.notes=[];self.scripts=0;self.options=[]
  def handle_starttag(self,tag,attrs):
   d=dict(attrs)
   if tag=='article':self.articles+=1
   if tag=='pre':self.pre=(d.get('class'),[])
   if tag=='select':self.fields.append(d)
   if tag=='option':assert 'selected' not in d;self.options.append(d)
   if tag=='input':assert not d.get('value')
   if tag=='textarea':self.textareas+=1;self.intextarea=True
   if tag=='script':self.scripts+=1
  def handle_data(self,t):
   if self.pre:self.pre[1].append(t)
   if self.intextarea:self.notes.append(t)
  def handle_endtag(self,t):
   if t=='textarea':self.intextarea=False
   if t=='pre':
    assert self.pre
    (self.prompts if self.pre[0]=='prompt' else self.responses).append(''.join(self.pre[1]));self.pre=None
 results={}
 expected_paired=sorted(rows,key=lambda r:(r['cohort'],r.get('family',0),r['id'],['W','D','P'].index(r['condition'])))
 for name,expected in [('HUMAN_REVIEW_PAIRED.html',expected_paired),('HUMAN_REVIEW_BLINDED.html',queue)]:
  p=Parse();p.feed((H/name).read_text())
  assert p.articles==72 and len(p.fields)==72*13 and p.textareas==72 and not p.notes and p.scripts==0
  assert p.prompts==[r['prompt'] for r in expected]
  assert p.responses==[r.get('raw_completion',r['response']) for r in expected]
  results[name]={'cards':p.articles,'blank_select_fields':len(p.fields),'blank_note_fields':p.textareas,'full_prompt_and_raw_response_roundtrip':True,'scripts':0}
 out=H/'runs/run-v1/output/model_audit_v1/results'
 source_checks={}
 for p in (out/'source').iterdir():
  if p.is_file():assert p.read_bytes()==(H/p.name).read_bytes(),p.name;source_checks[p.name]=sha(p)
 journal=ll(out/'generation_token_journal.jsonl');assert sum(r['event']=='start' for r in journal)==48
 responses=[r for r in rows if r['kind']=='safety'];clean_think=sum('</think' in r['response'] for r in responses)
 metadata=json.loads((out/'run_manifest.json').read_text());configs=[{k:v for k,v in d['generation_config'].items() if k!='transformers_version'} for d in metadata['conditions'].values()];assert all(d==configs[0] for d in configs)
 assert all(set(d['parameter_dtype_counts'])=={'torch.float16'} for d in metadata['conditions'].values())
 historical=json.loads((H/'historical_metric_audit.json').read_text());root=H.parents[1]
 assert all(sha(root/r['path'])==r['sha256'] for r in historical['conditions'].values())
 report={'raw_combined_records':72,'new_records':48,'reused_records':24,'frozen_agent_labels_unchanged':True,'earlier_historical_raw_files_unchanged':True,'review_builder_reproduces_all_bytes':True,'human_ratings_all_blank':True,'human_queue_omits_conditions_and_model_identity':True,'human_html':results,'source_copies_equal_preregistered_local_files':source_checks,'all_three_generation_defaults_equal_except_version_metadata':True,'all_model_parameters_fp16':True,'clean_safety_responses_containing_closed_think_tag':clean_think,'CoT_bug_not_needed_for_our_clean_text_rescore':clean_think==0,'artifact_hashes':before,'limitations':'No new weights, tokenizer/model execution or harmful functionality tests. Human blinding is imperfect as documented; structural masking does not prove reviewer ignorance.'}
 (H/'verification/final_artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['artifact_hashes','source_copies_equal_preregistered_local_files']},indent=2))
if __name__=='__main__':main()
