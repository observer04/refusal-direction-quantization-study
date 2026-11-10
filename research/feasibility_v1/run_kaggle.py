"""Bounded Kaggle-only runner. Built into a self-contained submitted script."""
import hashlib, importlib.metadata, json, os, random, signal, subprocess, sys, time, traceback
from pathlib import Path

assert Path('/kaggle').exists(), 'All model work is restricted to Kaggle.'
START=time.monotonic()
GENERATION_DEADLINE=6200
OUT=Path('/kaggle/working/feasibility_v1/results');OUT.mkdir(parents=True,exist_ok=True)
def sha(b): return hashlib.sha256(b).hexdigest()
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def elapsed(): return time.monotonic()-START
def say(*args): print(round(elapsed(),1),*args,flush=True)
def deadline(signum,frame): raise TimeoutError('Hard internal deadline reached')
signal.signal(signal.SIGALRM,deadline);signal.alarm(6500)
PINS=__PINS__
PROMPTS=__PROMPTS__
SOURCE_FILES=__SOURCE_FILES__
for name,source in SOURCE_FILES.items():
    target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source)
write('pins.json',PINS);write('prompts.json',PROMPTS)
manifest={'status':'setup','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'budget_seconds_external':6800,'generation_deadline_seconds':GENERATION_DEADLINE,
          'completion_cap':80,'global_user_completion_ceiling':160,'conditions':{},'failures':[],
          'quantization':{'method':'bitsandbytes native NF4','blocksize':64,'double_quant':False,
                          'compute_dtype':'float16','storage':'uint8','grid':'native refitted per checkpoint'},
          'pins':PINS,'system':'You are a helpful assistant.','thinking':False,'batch_size':1,'greedy':True}
def checkpoint():
    manifest['elapsed_seconds']=elapsed();write('run_manifest.json',manifest)
checkpoint()
if not os.environ.get('HF_TOKEN'):
    for p in Path('/kaggle/input').glob('**/phase1_hf_token.txt'):
        os.environ['HF_TOKEN']=p.read_text().strip();break
assert os.environ.get('HF_TOKEN'),'Private runtime credential unavailable'
os.environ['HF_HOME']='/kaggle/temp/feasibility_hf_cache'
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM']='false'
REQUIREMENTS='transformers==5.22.1\npeft==0.21.2\naccelerate==1.15.0\nbitsandbytes==0.50.2\nhuggingface-hub==1.33.0\nkernels==0.18.1\nsafetensors==0.8.0\n'
try:
    constraints=[]
    for name in ['torch','torchvision','torchaudio']:
        try:constraints.append(name+'=='+importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:pass
    assert any(x.startswith('torch==') for x in constraints)
    (OUT/'requirements-kaggle.txt').write_text(REQUIREMENTS)
    (OUT/'preserve-kaggle-torch.txt').write_text('\n'.join(constraints)+'\n')
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(OUT/'requirements-kaggle.txt'),'-c',str(OUT/'preserve-kaggle-torch.txt')],check=True,timeout=480)
    import gc
    import torch
    import bitsandbytes as bnb
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from transformers.generation.streamers import BaseStreamer
    from peft import PeftModel
    torch.set_grad_enabled(False);torch.manual_seed(0);random.seed(0)
    assert torch.cuda.device_count()==2,'Requires one T4 x2 session'
    manifest['gpus']=[{'name':torch.cuda.get_device_name(i),'memory':torch.cuda.get_device_properties(i).total_memory} for i in range(2)]
    assert all('T4' in r['name'] for r in manifest['gpus'])
    manifest['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','bitsandbytes','huggingface-hub','kernels','safetensors']}
    assert all(importlib.metadata.version(n)==v.split('==')[1] for v in constraints for n in [v.split('==')[0]])
    api=HfApi(token=os.environ['HF_TOKEN'])
    info=api.model_info(PINS['corrected']['id'],revision=PINS['corrected']['revision'])
    assert info.private and info.sha==PINS['corrected']['revision']
    adapter_dir=Path('/kaggle/temp/frozen_adapter');adapter_dir.mkdir(parents=True,exist_ok=True)
    import shutil
    for name,expected in PINS['adapter_sha256'].items():
        p=hf_hub_download(PINS['corrected']['id'],PINS['adapter_remote']+'/'+name,revision=PINS['corrected']['revision'],token=os.environ['HF_TOKEN'])
        assert sha(Path(p).read_bytes())==expected
        shutil.copyfile(p,adapter_dir/name)
    factors=load_file(str(adapter_dir/'adapter_model.safetensors'),device='cpu')
    assert len(factors)==128 and all(t.dtype==torch.float32 for t in factors.values())
    pairs={}
    for name,t in factors.items():
        prefix,which=name.rsplit('.lora_',1)
        module=prefix.removeprefix('base_model.model.')
        pairs.setdefault(module,{})[which.split('.')[0]]=t
    assert len(pairs)==64
    active=[name for name,ab in pairs.items() if torch.count_nonzero(ab['B']).item()]
    assert len(active)==42
    manifest['adapter']={'pairs':64,'active':42,'dtype':'float32','tensor_bytes':sum(t.numel()*t.element_size() for t in factors.values()),
                         'modules':sorted(pairs),'active_modules':sorted(active),'hashes':PINS['adapter_sha256'],'scaling':1.0}
    paths={}
    for label in ['original','corrected']:
        say('Downloading pinned checkpoint on Kaggle:',label)
        paths[label]=snapshot_download(PINS[label]['id'],revision=PINS[label]['revision'],token=os.environ['HF_TOKEN'],max_workers=4,
                                      allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model'])
    tokenizer=AutoTokenizer.from_pretrained(paths['original'],trust_remote_code=False)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=False,
                            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_storage=torch.uint8)
    write('quantization_config.json',config.to_dict())
    maxmem={0:'12GiB',1:'12GiB','cpu':'0GiB'}
    rows=[];expected_hashes={};selected=PROMPTS
    def tensor_hash(t):return sha(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    def load(label,quantized=False):
        say('Loading',label,'native NF4' if quantized else 'FP16')
        kwargs={'quantization_config':config} if quantized else {}
        model=AutoModelForImageTextToText.from_pretrained(paths[label],dtype=torch.float16,device_map='auto',max_memory=maxmem,trust_remote_code=False,**kwargs).eval()
        placements={k:str(v) for k,v in model.hf_device_map.items()}
        assert not any(v in ['cpu','disk'] for v in placements.values())
        layers=model.model.language_model.layers
        counts={'deltanet':sum(hasattr(l,'linear_attn') for l in layers),'attention':sum(hasattr(l,'self_attn') for l in layers)}
        assert len(layers)==32 and counts=={'deltanet':24,'attention':8},counts
        native=[]
        if quantized:
            for n,m in model.named_modules():
                if isinstance(m,bnb.nn.Linear4bit):
                    assert isinstance(m.weight,bnb.nn.Params4bit) and m.weight.dtype==torch.uint8 and m.weight.quant_state is not None,n
                    assert m.weight.quant_state.quant_type=='nf4' and m.weight.quant_state.blocksize==64
                    native.append(n)
            assert len(native)>100
            assert all(isinstance(model.get_submodule(n),bnb.nn.Linear4bit) for n in pairs), 'Unpacked writer module'
        manifest['conditions'][('Q' if quantized else '')+label]={'device_map':placements,'layer_counts':counts,
                         'native_linear4bit_count':len(native),'native_linear4bit_names':native,
                         'parameter_dtype_counts':{str(d):sum(p.numel() for p in model.parameters() if p.dtype==d) for d in {p.dtype for p in model.parameters()}},
                         'memory_allocated':[torch.cuda.memory_allocated(i) for i in range(2)]}
        checkpoint();return model
    def generate(model,condition,records):
        outpath=OUT/(condition+'_responses.jsonl')
        with outpath.open('a') as handle:
            for record in records:
                if elapsed()>GENERATION_DEADLINE:raise TimeoutError('Generation deadline')
                assert len(rows)<80
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
                checkpoint_label='original' if condition in ['W','QW','QW_D'] else 'corrected'
                row=dict(record,condition=condition,response=tokenizer.decode(ids,skip_special_tokens=True),
                         raw_completion=tokenizer.decode(ids,skip_special_tokens=False),rendered_prompt=rendered,
                         raw_output=tokenizer.decode(inputs['input_ids'][0].cpu().tolist()+ids,skip_special_tokens=False),generated_token_ids=ids,
                         generated_tokens=len(ids),eos_seen=terminal is not None,hit_token_limit=hit_cap,
                         stop_reason='generation_error' if generation_error else ('eos' if terminal is not None else ('length' if hit_cap else 'time_or_other')),
                         generation_error_type=type(generation_error).__name__ if generation_error else None,
                         seed=0,model=PINS[checkpoint_label],run='feasibility-v1',
                         generation_settings={'do_sample':False,'enable_thinking':False,'max_new_tokens':record['max_new_tokens'],
                                              'eos_token_id':eos,'pad_token_id':tokenizer.pad_token_id,'batch_size':1},
                         duration_seconds=time.monotonic()-started,input_rendered_sha256=sha(rendered.encode()),input_tokens=inputs['input_ids'].shape[1])
                rows.append(row);handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
                manifest['completed_responses']=len(rows);checkpoint()
                say(condition,'response',len([r for r in rows if r['condition']==condition]),'tokens',len(ids),'stop',row['stop_reason'])
                if generation_error:raise generation_error
                del output,inputs
    def free(model):
        del model;gc.collect();torch.cuda.empty_cache()
    def pack_hash(model):
        result={}
        for n,m in model.named_modules():
            if isinstance(m,bnb.nn.Linear4bit):
                qs=m.weight.quant_state
                result[n]={'packed':tensor_hash(m.weight),'absmax':tensor_hash(qs.absmax),'code':tensor_hash(qs.code),'shape':list(qs.shape)}
        return result
    # First family supplies a compute profile and counts toward the evaluation.
    model=load('original')
    for n,ab in pairs.items():
        w=model.get_submodule(n).weight
        merged=(w.float()+ab['B'].to(w.device)@ab['A'].to(w.device)).to(w.dtype)
        expected_hashes[n]=tensor_hash(merged)
        del merged
    write('expected_merged_writer_hashes.json',expected_hashes)
    generate(model,'W',PROMPTS[:4])
    profile=rows[:4];seconds_per_token=max(r['duration_seconds']/max(1,r['generated_tokens']) for r in profile)
    selected_families=0
    for nf in [4,3,2]:
        candidate=PROMPTS[:4*nf]
        upper_tokens=sum(r['max_new_tokens'] for r in candidate)*5-sum(r['generated_tokens'] for r in rows)
        estimate=upper_tokens*seconds_per_token*1.35+3*180+600
        if elapsed()+estimate<6800:
            selected_families=nf;break
    manifest['profile']={'durations':[r['duration_seconds'] for r in profile],'tokens':[r['generated_tokens'] for r in profile],
                         'worst_seconds_per_token':seconds_per_token,'inflation':1.35,'selected_family_prefix':selected_families,
                         'decision_uses_output_text':False,'estimated_remaining_seconds':estimate}
    write('profile_decision.json',manifest['profile']);checkpoint()
    selected=PROMPTS[:4*(selected_families or 2)]
    write('selected_prompts.json',selected)
    generate(model,'W',selected[4:]);del model;gc.collect();torch.cuda.empty_cache()
    model=load('corrected')
    checks={n:tensor_hash(model.get_submodule(n).weight)==h for n,h in expected_hashes.items()}
    manifest['frozen_delta_merge_verification']={'all_writer_hashes_match':all(checks.values()),'checks':checks,
                   'definition':'FP16(original.float + saved FP32 B@A); compare corrected pinned full tensor SHA-256'}
    write('frozen_delta_merge_verification.json',manifest['frozen_delta_merge_verification'])
    generate(model,'D',selected);del model;gc.collect();torch.cuda.empty_cache()
    manifest['baseline_complete']=True;checkpoint()
    if not selected_families:raise RuntimeError('Budget profile supports baseline only; quantization not launched')
    assert all(checks.values()),'Frozen adapter does not exactly reproduce corrected FP16 writers; stop quantization'
    # Do not launch quantization if conservative remaining-time estimate fails after baseline.
    remaining_est=sum(r['max_new_tokens'] for r in selected)*3*seconds_per_token*1.35+2*180+600
    assert elapsed()+remaining_est<6800,'Insufficient profiled budget for all quantization arms'
    model=load('original',True)
    packed_before=pack_hash(model);write('QW_native_state_hashes.json',packed_before)
    generate(model,'QW',selected)
    # Load actual factor bytes, never invoke abliterate() or calculate v^T Q(W).
    model=PeftModel.from_pretrained(model,str(adapter_dir),is_trainable=False,autocast_adapter_dtype=True).eval()
    numerical=[]
    for n,ab in pairs.items():
        m=model.base_model.model.get_submodule(n)
        assert m.scaling['default']==1.0 and m.lora_A['default'].weight.dtype==torch.float32 and m.lora_B['default'].weight.dtype==torch.float32
        assert torch.equal(m.lora_A['default'].weight.cpu(),ab['A']) and torch.equal(m.lora_B['default'].weight.cpu(),ab['B'])
        x=torch.linspace(-0.25,0.25,m.in_features,device=m.weight.device,dtype=torch.float16).view(1,1,-1)
        y=m(x);base=m.base_layer(x)
        expected=(base.float()+(x.float()@ab['A'].to(x.device).T)@ab['B'].to(x.device).T).to(base.dtype)
        err=(y.float()-expected.float()).abs().max().item()
        tolerance=0.004*max(1.0,expected.float().abs().max().item())
        assert err<=tolerance,(n,err,tolerance)
        numerical.append({'module':n,'max_abs_error':err,'tolerance':tolerance,'exact_saved_factors':True,'base_native_packed':isinstance(m.base_layer,bnb.nn.Linear4bit)})
    packed_after={n.removeprefix('base_model.model.').removesuffix('.base_layer'):v for n,v in pack_hash(model).items()}
    assert packed_before==packed_after,'Quantized backbone changed when loading frozen adapter'
    write('QW_D_adapter_verification.json',{'all64_factors_exact':True,'same_native_packed_state':True,'module_checks':numerical,
         'adapter_storage_bytes':manifest['adapter']['tensor_bytes'],'adapter_dtype':'float32','rank':1,'scaling':1,
         'float_reference_limitation':'Low-rank factor evaluation and output cast; original FP16 merged delta includes per-entry rounding not representable exactly by rank one.'})
    generate(model,'QW_D',selected)
    del model;gc.collect();torch.cuda.empty_cache()
    model=load('corrected',True)
    write('QD_native_state_hashes.json',pack_hash(model))
    generate(model,'QD',selected)
    del model;gc.collect();torch.cuda.empty_cache()
    manifest['status']='five_conditions_complete'
except Exception as exc:
    # Exception text is restricted to model/library errors, never credentials or HTTP headers.
    manifest['status']='stopped';manifest['failures'].append({'type':type(exc).__name__,'reason':str(exc)[:1500],'elapsed_seconds':elapsed()})
    say('Stopped:',type(exc).__name__,str(exc)[:600])
    traceback.print_exc(limit=5)
finally:
    signal.alarm(0);checkpoint()
    # Persist only reviewed source and small output files to a new private experiment path.
    try:
        if 'api' in globals():
            info=api.model_info(PINS['corrected']['id']);assert info.private
            token=os.environ['HF_TOKEN'].encode()
            for p in OUT.rglob('*'):
                if p.is_file():assert token not in p.read_bytes() and p.stat().st_size<20_000_000
            commit=api.upload_folder(repo_id=PINS['corrected']['id'],folder_path=str(OUT),path_in_repo='research/feasibility_v1/run-v1',
                commit_message='Add bounded fresh semantic and native NF4 feasibility evidence')
            write('persistence.json',{'private':True,'commit':commit.oid,'path':'research/feasibility_v1/run-v1','repo':PINS['corrected']['id']})
            say('Private evidence commit',commit.oid)
    except Exception as exc:
        write('persistence_failure.json',{'type':type(exc).__name__,'reason':str(exc)[:500]})
        say('Evidence remains in Kaggle outputs; private upload failed:',type(exc).__name__)
    say('Finished status',manifest['status'],'completions',manifest.get('completed_responses',0))
