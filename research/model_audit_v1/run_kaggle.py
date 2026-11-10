"""Independent bounded FP16 audit. Execute only on Kaggle."""
import hashlib,importlib.metadata,json,os,random,signal,subprocess,sys,time,traceback
from pathlib import Path
assert Path('/kaggle').exists(),'All model work is Kaggle-only'
START=time.monotonic();GENERATION_DEADLINE=6100
OUT=Path('/kaggle/working/model_audit_v1/results');OUT.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def elapsed():return time.monotonic()-START
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def say(*a):print(round(elapsed(),1),*a,flush=True)
def deadline(s,f):raise TimeoutError('Internal audit deadline')
signal.signal(signal.SIGALRM,deadline);signal.alarm(6500)
PINS=__PINS__;MATCHED=__MATCHED__;FRESH=__FRESH__;SOURCE_FILES=__SOURCE_FILES__
for n,s in SOURCE_FILES.items():
 p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
for r in MATCHED:r['cohort']='matched'
for r in FRESH:r['cohort']='fresh'
write('pins.json',PINS);write('matched_prompts.json',MATCHED);write('fresh_fixture.json',FRESH)
manifest={'status':'setup','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'budget_seconds_external':6800,'generation_deadline_seconds':6100,'completion_cap':84,'conditions':{},'failures':[],'pins':PINS,'system':'You are a helpful assistant.','thinking':False,'batch_size':1,'greedy':True}
def checkpoint():manifest['elapsed_seconds']=elapsed();write('run_manifest.json',manifest)
checkpoint()
if not os.environ.get('HF_TOKEN'):
 for p in Path('/kaggle/input').glob('**/phase1_hf_token.txt'):os.environ['HF_TOKEN']=p.read_text().strip();break
assert os.environ.get('HF_TOKEN'),'Private runtime credential unavailable'
os.environ['HF_HOME']='/kaggle/temp/audit_hf_cache';os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True';os.environ['TOKENIZERS_PARALLELISM']='false'
REQUIREMENTS='transformers==5.22.1\npeft==0.21.2\naccelerate==1.15.0\nbitsandbytes==0.50.2\nhuggingface-hub==1.33.0\nkernels==0.18.1\nsafetensors==0.8.0\n'
try:
 constraints=[]
 for n in ['torch','torchvision','torchaudio']:
  try:constraints.append(n+'=='+importlib.metadata.version(n))
  except importlib.metadata.PackageNotFoundError:pass
 assert any(x.startswith('torch==') for x in constraints)
 (OUT/'requirements-kaggle.txt').write_text(REQUIREMENTS);(OUT/'preserve-kaggle-torch.txt').write_text('\n'.join(constraints)+'\n')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,stdout=subprocess.DEVNULL)
 subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(OUT/'requirements-kaggle.txt'),'-c',str(OUT/'preserve-kaggle-torch.txt')],check=True,timeout=480)
 import gc,torch
 from huggingface_hub import HfApi,snapshot_download
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 from transformers.generation.streamers import BaseStreamer
 torch.set_grad_enabled(False);torch.manual_seed(0);random.seed(0)
 assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
 manifest['gpus']=[{'name':torch.cuda.get_device_name(i),'memory':torch.cuda.get_device_properties(i).total_memory} for i in range(2)]
 manifest['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','bitsandbytes','huggingface-hub','kernels','safetensors']}
 assert all(importlib.metadata.version(x.split('==')[0])==x.split('==')[1] for x in constraints)
 api=HfApi(token=os.environ['HF_TOKEN']);info=api.model_info(PINS['corrected']['id'],revision=PINS['corrected']['revision']);assert info.private and info.sha==PINS['corrected']['revision']
 paths={}
 tokenpath=snapshot_download(PINS['original']['id'],revision=PINS['original']['revision'],token=os.environ['HF_TOKEN'],allow_patterns=['*.json','*.jinja','*.txt','*.model'],ignore_patterns=['model.safetensors.index.json'])
 tokenizer=AutoTokenizer.from_pretrained(tokenpath,trust_remote_code=False);tokenizer.padding_side='left'
 if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
 manifest['tokenizer_hashes']={p.name:sha(p.read_bytes()) for p in Path(tokenpath).glob('*') if p.is_file() and p.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']}
 maxmem={0:'12GiB',1:'12GiB','cpu':'0GiB'};rows=[]
 def load(label):
  say('Downloading pinned',label)
  paths[label]=snapshot_download(PINS[label]['id'],revision=PINS[label]['revision'],token=os.environ['HF_TOKEN'],max_workers=4,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model'])
  say('Loading',label,'FP16')
  model=AutoModelForImageTextToText.from_pretrained(paths[label],dtype=torch.float16,device_map='auto',max_memory=maxmem,trust_remote_code=False).eval()
  placements={k:str(v) for k,v in model.hf_device_map.items()};assert not any(v in ['cpu','disk'] for v in placements.values())
  layers=model.model.language_model.layers;assert len(layers)==32 and sum(hasattr(l,'linear_attn') for l in layers)==24
  manifest['conditions'][label]={'device_map':placements,'generation_config':model.generation_config.to_dict(),'parameter_dtype_counts':{str(d):sum(p.numel() for p in model.parameters() if p.dtype==d) for d in {p.dtype for p in model.parameters()}}}
  checkpoint();return model
 def generate(model,condition,records):
     outpath=OUT/(condition+'_responses.jsonl')
     with outpath.open('a') as handle:
         for record in records:
             if elapsed()>GENERATION_DEADLINE:raise TimeoutError('Generation deadline')
             assert len(rows)<84
             messages=[{'role':'system','content':manifest['system']},{'role':'user','content':record['prompt']}]
             rendered=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
             inputs=tokenizer(rendered,return_tensors='pt',return_token_type_ids=False).to(model.device)
             eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
             eos_set=set(eos if isinstance(eos,list) else [eos])
             started=time.monotonic()
             class Journal(BaseStreamer):
                 def __init__(self):
                     self.first=True;self.ids=[]
                     self.handle=(OUT/'generation_token_journal.jsonl').open('a')
                     self.handle.write(json.dumps({'event':'start','condition':condition,'id':record['id'],'prompt':record['prompt'],
                         'rendered_prompt':rendered,'input_ids':inputs['input_ids'][0].cpu().tolist(),'max_new_tokens':record['max_new_tokens'],
                         'seed':0,'greedy':True,'elapsed_seconds':elapsed()})+'\n');self.handle.flush();os.fsync(self.handle.fileno())
                 def put(self,value):
                     if self.first:self.first=False;return
                     tokens=value.detach().cpu().reshape(-1).tolist();self.ids.extend(tokens)
                     self.handle.write(json.dumps({'event':'tokens','condition':condition,'id':record['id'],'token_ids':tokens})+'\n');self.handle.flush()
                     if len(self.ids)%32==0:os.fsync(self.handle.fileno())
                 def end(self):self.handle.flush();os.fsync(self.handle.fileno())
             journal=Journal();generation_error=None
             try:
                 with torch.inference_mode():
                     output=model.generate(**inputs,max_new_tokens=record['max_new_tokens'],do_sample=False,pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=eos,max_time=max(1,GENERATION_DEADLINE-elapsed()),renormalize_logits=False,streamer=journal)
                 ids=output[0,inputs['input_ids'].shape[1]:].detach().cpu().tolist()
                 assert ids==journal.ids
             except Exception as exc:
                 ids=journal.ids;generation_error=exc
             finally:
                 journal.end();journal.handle.close()
             terminal=next((i for i,t in enumerate(ids) if t in eos_set),None)
             if terminal is not None:ids=ids[:terminal+1]
             hit_cap=terminal is None and len(ids)>=record['max_new_tokens']
             checkpoint_label={'W':'original','D':'corrected','P':'publisher'}[condition]
             row=dict(record,condition=condition,response=tokenizer.decode(ids,skip_special_tokens=True),
                      raw_completion=tokenizer.decode(ids,skip_special_tokens=False),rendered_prompt=rendered,
                      raw_output=tokenizer.decode(inputs['input_ids'][0].cpu().tolist()+ids,skip_special_tokens=False),generated_token_ids=ids,
                      generated_tokens=len(ids),eos_seen=terminal is not None,hit_token_limit=hit_cap,
                      stop_reason='generation_error' if generation_error else ('eos' if terminal is not None else ('length' if hit_cap else 'time_or_other')),
                      generation_error_type=type(generation_error).__name__ if generation_error else None,
                      seed=0,model=PINS[checkpoint_label],run='model-audit-v1',
                      generation_settings={'do_sample':False,'enable_thinking':False,'max_new_tokens':record['max_new_tokens'],
                                           'eos_token_id':eos,'pad_token_id':tokenizer.pad_token_id,'batch_size':1},
                      duration_seconds=time.monotonic()-started,input_rendered_sha256=sha(rendered.encode()),input_tokens=inputs['input_ids'].shape[1])
             rows.append(row);handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
             manifest['completed_responses']=len(rows);checkpoint()
             say(condition,'response',len([r for r in rows if r['condition']==condition]),'tokens',len(ids),'stop',row['stop_reason'])
             if generation_error:raise generation_error
             del output,inputs
 # Prove exact matched prompt rendering and settings before reusing saved W/D.
 expected=json.loads(SOURCE_FILES['matched_reuse_metadata.json'])
 assert manifest['packages']==expected['packages'],('Package mismatch',manifest['packages'],expected['packages'])
 model=load('publisher')
 eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
 reuse=[]
 for record in MATCHED:
  rendered=tokenizer.apply_chat_template([{'role':'system','content':manifest['system']},{'role':'user','content':record['prompt']}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
  ids=tokenizer(rendered,return_token_type_ids=False)['input_ids']
  e=expected['records'][record['id']]
  assert sha(rendered.encode())==e['input_rendered_sha256'] and ids==e['input_ids']
  settings={'do_sample':False,'enable_thinking':False,'max_new_tokens':record['max_new_tokens'],'eos_token_id':eos,'pad_token_id':tokenizer.pad_token_id,'batch_size':1}
  assert settings==e['generation_settings'],(record['id'],'Generation settings mismatch')
  reuse.append({'id':record['id'],'exact_rendered_prompt_and_input_ids':True,'exact_explicit_generation_settings':True})
 write('matched_reuse_verification.json',{'all':True,'records':reuse,'baseline_files':expected['files'],'limitation':'Same saved W/D baseline reused; architecture default generation configuration retained and saved for direct audit.'})
 generate(model,'P',MATCHED)
 generate(model,'P',FRESH[:6])
 profile=rows[-6:];spt=max(r['duration_seconds']/max(1,r['generated_tokens']) for r in profile);nf=0
 for f in [4,3,2]:
  candidate=FRESH[:6*f];remaining_tokens=sum(r['max_new_tokens'] for r in candidate)*3-sum(r['generated_tokens'] for r in profile)
  estimate=remaining_tokens*spt*1.25+2*240
  if elapsed()+estimate<GENERATION_DEADLINE:nf=f;break
 decision={'selected_family_prefix':nf,'worst_seconds_per_token':spt,'inflation':1.25,'profile_token_counts':[r['generated_tokens'] for r in profile],'profile_durations':[r['duration_seconds'] for r in profile],'elapsed_seconds':elapsed(),'estimated_remaining_seconds':estimate,'output_text_inspected':False}
 write('profile_decision.json',decision);manifest['profile']=decision;checkpoint()
 if not nf:raise RuntimeError('Fresh three-model validation does not fit; preserve publisher matched and profile evidence')
 selected=FRESH[:6*nf];write('selected_fresh_prompts.json',selected)
 generate(model,'P',selected[6:]);del model;gc.collect();torch.cuda.empty_cache()
 model=load('original');generate(model,'W',selected);del model;gc.collect();torch.cuda.empty_cache()
 model=load('corrected');generate(model,'D',selected);del model;gc.collect();torch.cuda.empty_cache()
 manifest['status']='all_selected_conditions_complete'
except Exception as exc:
 manifest['status']='stopped';manifest['failures'].append({'type':type(exc).__name__,'reason':str(exc)[:1000],'elapsed_seconds':elapsed()});say('Stopped',type(exc).__name__,str(exc)[:500]);traceback.print_exc(limit=5)
finally:
 signal.alarm(0);checkpoint()
 try:
  if 'api' in globals():
   info=api.model_info(PINS['corrected']['id']);assert info.private
   token=os.environ['HF_TOKEN'].encode()
   for p in OUT.rglob('*'):
    if p.is_file():assert token not in p.read_bytes() and p.stat().st_size<20_000_000
   commit=api.upload_folder(repo_id=PINS['corrected']['id'],folder_path=str(OUT),path_in_repo='research/model_audit_v1/run-v1',commit_message='Preserve bounded independent FP16 audit completions and source')
   write('persistence.json',{'private':True,'commit':commit.oid,'path':'research/model_audit_v1/run-v1','repo':PINS['corrected']['id']});say('Private evidence commit',commit.oid)
 except Exception as exc:
  write('persistence_failure.json',{'type':type(exc).__name__,'reason':str(exc)[:500]});say('Private evidence upload failed',type(exc).__name__)
 say('Finished',manifest['status'],'completions',manifest.get('completed_responses',0))
