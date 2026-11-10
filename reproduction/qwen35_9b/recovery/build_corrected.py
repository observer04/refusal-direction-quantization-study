"""Build a fresh-extraction corrected notebook; preserve the original failed replay."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
nb=copy.deepcopy(json.loads((ROOT/'Reproduce_Qwen3_5_9B_Heretic.ipynb').read_text()))
def source(index):return ''.join(nb['cells'][index]['source'])
def replace(index,old,new):
    text=source(index)
    assert text.count(old)==1,(index,old)
    nb['cells'][index]['source']=text.replace(old,new).splitlines(keepends=True)

intro='''# Qwen3.5-9B: corrected projected-direction reproduction

Run on Kaggle T4 x2 in a fresh session. This notebook extracts fresh means from400 harmful and400 harmless prompts, orthogonalizes each refusal vector against the corresponding harmless mean, normalizes each vector, THEN interpolates slots19/20 at.49 and applies the fixed published weights. No search or quantization.

The omitted `orthogonalize_direction` processing was identified through weight-delta forensics and a raw-versus-projected causal pilot. The author did not publish that flag, exact environment or unrounded parameters; this remains a measured approximate FP16 reconstruction rather than a claim of byte-identical author provenance.

Historical100-prompt marker counts and benign first-token KL are retained for comparability. They do not establish harmful compliance, full HarmBench performance or general capability retention. Save/reload verification precedes private Hub persistence.
'''
nb['cells'][0]['source']=intro.splitlines(keepends=True)
for index,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code':
        cell['source']=source(index).replace('qwen35_reproduction','qwen35_corrected').splitlines(keepends=True)
replace(3,'orthogonalize_direction=False','orthogonalize_direction=True')
replace(5,'refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)',
'''raw_refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
good_directions = F.normalize(good_means, p=2, dim=1)
projection = (raw_refusal_directions * good_directions).sum(dim=1, keepdim=True)
refusal_directions = F.normalize(raw_refusal_directions - projection * good_directions, p=2, dim=1)
manifest['direction_processing'] = 'per-slot harmless-mean orthogonalization, normalize, then interpolate'
manifest['orthogonalize_direction'] = True
manifest['historical_test_exposed'] = True''')
replace(5,"'directions': refusal_directions}","'directions': refusal_directions, 'raw_directions': raw_refusal_directions, 'orthogonalize_direction': True}")
replace(5,"model.model.save_pretrained(OUT / 'adapter', safe_serialization=True)",
'''model.model.save_pretrained(OUT / 'adapter', safe_serialization=True)
adapter_config_path = OUT / 'adapter/adapter_config.json'
adapter_config = json.loads(adapter_config_path.read_text())
adapter_config['base_model_name_or_path'] = PROVENANCE['base_model']['id']
adapter_config['revision'] = PROVENANCE['base_model']['revision']
adapter_config_path.write_text(json.dumps(adapter_config, indent=2))''')
replace(7,'/kaggle/temp/Qwen3.5-9B-heretic-replay','/kaggle/temp/Qwen3.5-9B-heretic-corrected')
replace(9,"'/Qwen3.5-9B-heretic-replay-20261107'","'/Qwen3.5-9B-heretic-corrected-20261109'")
replace(9,'# Qwen3.5-9B published-parameter replay','# Qwen3.5-9B corrected projected-direction reproduction')
replace(9,'Best-effort FP16 replay of trohrbaugh/Qwen3.5-9B-heretic-v2 rounded settings.',
        'Fresh-extraction FP16 reconstruction with harmless-mean-orthogonalized refusal directions and fixed published settings.')
replace(2,"'reproduction_scope': 'rounded published-parameter replay, not exact weight reproduction'",
        "'reproduction_scope': 'fresh projected-direction reconstruction, not byte-identical author weights'")
replace(6,"assert np.isfinite(kl), 'Non-finite KL; stop this run.'",'''assert np.isfinite(kl), 'Non-finite KL; stop this run.'
per_prompt_kl = (baseline_logprobs.exp() * (baseline_logprobs - edited_logprobs)).sum(dim=-1)
assert abs(per_prompt_kl.mean().item() - kl) < 1e-5
(OUT / 'benign_first_token_kl_terms.json').write_text(json.dumps({
    'definition':'sum_v exp(baseline_logp_v) * (baseline_logp_v - edited_logp_v)',
    'dataset':'mlabonne/harmless_alpaca', 'split':'test[:100]',
    'per_prompt_kl':per_prompt_kl.tolist(), 'mean':per_prompt_kl.mean().item()}, indent=2))''')
replace(8,"manifest['status'] = 'reload_verified'","manifest['status'] = 'reload_compared'")
replace(8,"assert response_matches == 100, 'Reload differs from pre-save responses; inspect before claiming verification.'",
'''assert response_matches == 100, 'Reload differs from pre-save responses; inspect before claiming verification.'
manifest['status'] = 'reload_verified'
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))''')

# Record the rendered evaluation prefix rather than relying on a prose setting.
replace(3,"print('Target modules:', dict(components))",'''print('Target modules:', dict(components))
rendered_probe = model.tokenizer.apply_chat_template(
    [[{'role':'system','content':settings.system_prompt},{'role':'user','content':'What is 1+1?'}]],
    add_generation_prompt=True, tokenize=False)[0]
probe_ids = model.tokenizer(rendered_probe, return_token_type_ids=False)['input_ids']
manifest['evaluation_prompt_probe'] = {'rendered_sha256':hashlib.sha256(rendered_probe.encode()).hexdigest(),
                                       'last32_token_ids':probe_ids[-32:]}''')

# Freeze the same12fresh prompts already chosen for the causal pilot's long sample.
# The fixture is populated only after the pilot completes; missing fixtures stop a run.
fixture_path=HERE/'paired_long_prompts.json'
paired_records=json.loads(fixture_path.read_text()) if fixture_path.exists() else []
helper=(HERE/'evaluation.py').read_text()
paired_setup='''
# Fixed paired longer-answer diagnostic; this is not a population benchmark.
import importlib.util
EVALUATION_SOURCE = __HELPER_SOURCE__
PAIRED_LONG_RECORDS = __PAIRED_RECORDS__
assert len(PAIRED_LONG_RECORDS)==12, 'Populate fixed paired fixture from the completed causal pilot.'
(OUT / 'evaluation.py').write_text(EVALUATION_SOURCE)
spec=importlib.util.spec_from_file_location('corrected_evaluation',OUT/'evaluation.py')
ev=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
(OUT / 'paired_long_prompts.json').write_text(json.dumps(PAIRED_LONG_RECORDS,indent=2))
manifest['paired_long_prompt_hashes']=[ev.digest(row['prompt']) for row in PAIRED_LONG_RECORDS]
def run_paired_long(condition):
    rows=[]
    path=OUT/(condition+'_paired_long_responses.jsonl')
    print('Starting paired long diagnostic:',condition,flush=True)
    with path.open('w') as handle:
        for start in range(0,len(PAIRED_LONG_RECORDS),BATCH_SIZE):
            batch_rows=ev.generate_rows(model,PAIRED_LONG_RECORDS[start:start+BATCH_SIZE],
                lambda text:Prompt(system=settings.system_prompt,user=text),condition=condition,
                split='paired_long_diagnostic',batch_size=BATCH_SIZE,max_new_tokens=512)
            rows.extend(batch_rows)
            handle.write(''.join(json.dumps(row)+'\\n' for row in batch_rows))
            handle.flush()
            print('Paired long diagnostic:',condition,len(rows),'/',len(PAIRED_LONG_RECORDS),flush=True)
    return rows
original_long=run_paired_long('original')
print('Original paired long diagnostic:',ev.summarize_rows(original_long),flush=True)
'''.replace('__HELPER_SOURCE__',repr(helper)).replace('__PAIRED_RECORDS__',repr(paired_records))
nb['cells'][4]['source']=(source(4)+'\n'+paired_setup).splitlines(keepends=True)

# Keep the main checkpoint durable before spending the final diagnostic tokens.
paired_finish='''
# Paired longer answers on the freshly reloaded corrected weights.
corrected_long=run_paired_long('corrected_reloaded')
summary['paired_long_diagnostic']={'max_new_tokens':512,'n_prompts':12,
    'original':ev.summarize_rows(original_long),'corrected':ev.summarize_rows(corrected_long),
    'scope':'Paired qualitative diagnostic; no population harmful-compliance or capability claim'}
queue,mapping=ev.make_blind_review_queue(original_long+corrected_long,n_prompts=12)
for name,rows in [('paired_long_blind_review_queue',queue),('paired_long_blind_review_mapping',mapping)]:
    (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\\n' for row in rows))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
manifest['status']='paired_diagnostics_generated'
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
api.upload_folder(repo_id=repo_id,folder_path=str(OUT),path_in_repo='experiments',
    commit_message='Add fixed paired longer-answer diagnostic and review queue')
manifest['status']='persisted_with_paired_diagnostics'
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
api.upload_file(repo_id=repo_id,path_or_fileobj=str(OUT/'run_manifest.json'),
    path_in_repo='experiments/run_manifest.json',commit_message='Confirm paired diagnostic persistence')
print('Paired diagnostic persisted:',json.dumps(summary['paired_long_diagnostic']),flush=True)
'''
nb['cells'].insert(10,{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                      'source':paired_finish.splitlines(keepends=True)})
for cell in nb['cells']:
    text=''.join(cell['source'])
    if cell['cell_type']=='code':
        ast.parse(text)
        cell['execution_count']=None
        cell['outputs']=[]
    cell['id']=hashlib.sha256(text.encode()).hexdigest()[:12]
dest=HERE/'kaggle-corrected'
dest.mkdir(exist_ok=True)
for path in [ROOT/'Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb',
             HERE/'Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb',
             dest/'Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb']:
    path.write_text(json.dumps(nb,indent=1))
(dest/'kernel-metadata.json').write_text(json.dumps({
    'id':'ommprakash007/qwen35-recovery-corrected','title':'Qwen35 Recovery Corrected',
    'code_file':'Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb','language':'python','kernel_type':'notebook',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'dataset_sources':['ommprakash007/qwen35-phase1-private-runtime-input'],
    'kernel_sources':[],'competition_sources':[],'model_sources':[]},indent=2))
print('Corrected notebook prepared; not submitted. Fixed paired fixtures:',len(paired_records))
