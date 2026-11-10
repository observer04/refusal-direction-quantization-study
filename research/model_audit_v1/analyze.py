"""Rebuild descriptive counts and paired case tables from frozen agent labels."""
import ast,collections,csv,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
def ll(n):return [json.loads(s) for s in (H/n).read_text().splitlines() if s.strip()]
def main():
 data={(r['cohort'],r['condition'],r['id']):r for r in ll('all_responses.jsonl')}
 labeled=[]
 for prefix in ['saved','new']:
  freeze=json.loads((H/f'verification/{prefix}_labels_frozen_before_unmask.json').read_text())
  labelname=prefix+'_blind_labels.jsonl'
  assert hashlib.sha256((H/labelname).read_bytes()).hexdigest()==freeze['labels_sha256']
  labels={r['blind_id']:r for r in ll(labelname)}
  for m in ll(prefix+'_blind_mapping.jsonl'):
   cohort=m.get('cohort','matched');r=data[cohort,m['condition'],m['id']]
   labeled.append(dict(r,label=labels[m['blind_id']]))
 assert len(labeled)==len(data) and len({(r['cohort'],r['condition'],r['id']) for r in labeled})==len(data)
 (H/'agent_labeled_responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labeled))
 groups=collections.defaultdict(list)
 for r in labeled:groups[r['cohort'],r['condition']].append(r)
 summary={}
 for (cohort,c),rs in sorted(groups.items()):
  safety=[r for r in rs if r['kind']=='safety'];benign=[r for r in rs if r['kind']!='safety'];utility=[r for r in benign if r['kind']=='utility']
  summary[cohort+' | '+c]={'n':len(rs),'safety_n':len(safety),'fulfillment':dict(collections.Counter(r['label']['requested_fulfillment'] for r in safety)),'explicit_refusal':dict(collections.Counter(r['label']['explicit_refusal'] for r in safety)),'safety_withholding':dict(collections.Counter(r['label']['safety_withholding'] for r in safety)),'safety_capped':sum(r['hit_token_limit'] for r in safety),'all_capped':sum(r['hit_token_limit'] for r in rs),'benign_n':len(benign),'benign_useful':dict(collections.Counter(r['label']['benign_useful'] for r in benign)),'benign_overrefusal':dict(collections.Counter(r['label']['overrefusal'] for r in benign)),'utility_n':len(utility),'numerical_correct':dict(collections.Counter(r['label']['numerical_correct'] for r in utility))}
 paired=collections.defaultdict(dict)
 for r in labeled:
  if r['kind']=='safety':paired[r['cohort'],r['id']][r['condition']]=r
 contrasts=[]
 lines=['# Independent paired semantic review','', 'W = original; D = reconstruction; P = actual publisher. Each cell: requested fulfillment / safety withholding / cap. These are observed-content labels, with harmful accuracy unverified. One agent reviewed condition-masked queues; prior exposure and output style limit masking. Human ratings remain blank.','']
 for cohort in ['matched','fresh']:
  lines+=['','## '+cohort.capitalize(),'','| Case | W | D | P | Item interpretation |','|---|---|---|---|---|']
  for (ch,ident),arms in sorted(paired.items()):
   if ch!=cohort:continue
   cells=[];notes=[]
   for c in ['W','D','P']:
    r=arms.get(c)
    if not r:cells.append('missing');continue
    lab=r['label'];cells.append(lab['requested_fulfillment']+' / '+lab['safety_withholding']+' / '+str(r['hit_token_limit']))
    notes.append(c+': '+lab['reason'].replace('|','/').replace('\n',' '))
   lines.append('|'+ident+'|'+'|'.join(cells)+'|'+'<br>'.join(notes)+'|')
   contrasts.append({'cohort':cohort,'id':ident,**{c:{'fulfillment':r['label']['requested_fulfillment'],'withholding':r['label']['safety_withholding'],'explicit_refusal':r['label']['explicit_refusal'],'fact_correction':r['label']['factual_correction'],'knowledge_gap':r['label']['inability_knowledge_gap'],'ambiguous_request':r['label']['ambiguous_request'],'length_cap':r['hit_token_limit']} for c,r in arms.items()}})
 lines+=['','Full raw text is in [paired human review](HUMAN_REVIEW_PAIRED.html); condition-masked human fields and mapping are separate. Fact correction and missing contacts are not automatically safety refusal. Capped negative/partial prefixes do not settle unseen continuations.']
 (H/'ITEM_REVIEW.md').write_text('\n'.join(lines)+'\n')
 (H/'semantic_summary.json').write_text(json.dumps({'scope':'Descriptive selected-case counts; no population rate, confidence interval, verified harmful efficacy or independent human consensus.','groups':summary,'pairs':contrasts},indent=2)+'\n')
 source=(H/'reference/classifier_initial.py').read_text();allowed=[]
 for n in ast.parse(source).body:
  if isinstance(n,ast.ImportFrom) and n.module in ['__future__','src']:continue
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign,ast.AnnAssign)):allowed.append(n)
 ns={'__name__':'audit_v3','RESULTS_VERSION':'public_source_snapshot'};exec(compile(ast.Module(body=allowed,type_ignores=[]),'<public_v3>','exec'),ns);assert ns['VERSION']=='3.0'
 markers=json.loads((H/'historical_metric_audit.json').read_text())['markers']
 classifier=[]
 for r in labeled:
  if r['kind']!='safety':continue
  lab=r['label'];t=' '.join(r['response'].lower().replace('*','').replace('’',"'").split());marker=not t or any(p.lower() in t for p in markers)
  ref,layer,reason,confidence=ns['classify_item']({'response':r['response'],'reasoning':'','behavior_id':r['id']})
  classifier.append({'cohort':r['cohort'],'condition':r['condition'],'id':r['id'],'upstream_marker_refusal':marker,'public_v3_is_refusal_without_overrides':ref,'public_v3_layer':layer,'public_v3_reason':reason,'public_v3_confidence':confidence,'independent_fulfillment':lab['requested_fulfillment'],'independent_safety_withholding':lab['safety_withholding'],'independent_explicit_refusal':lab['explicit_refusal'],'independent_fact_correction':lab['factual_correction'],'length_cap':r['hit_token_limit']})
 (H/'classifier_comparison_all.json').write_text(json.dumps({'scope':'Exact public v3 source on OUR nonthinking outputs with empty reasoning and NO external overrides; not external 400-run reproduction. Independent labels frozen before this comparison.','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'rows':classifier},indent=2)+'\n')
 with (H/'classifier_comparison_all.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(classifier[0]));w.writeheader();w.writerows(classifier)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
