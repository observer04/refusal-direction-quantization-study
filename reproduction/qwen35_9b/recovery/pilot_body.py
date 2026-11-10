"""Recovery pilot body, appended to the pinned Kaggle setup by build_pilot.py.

This file is not a standalone local model runner. All model work is on Kaggle.
"""
import importlib.util
import math
import shutil
from huggingface_hub import hf_hub_download

assert Path('/kaggle').exists()
audit_paths = list(Path('/kaggle/input').glob('**/recovery_audit/summary.json'))
assert len(audit_paths) <= 1, audit_paths
fingerprints=[]
if audit_paths:
    AUDIT_DIR=audit_paths[0].parent
    AUDIT=json.loads(audit_paths[0].read_text())
    assert AUDIT.get('complete') is True
    recovered=torch.load(AUDIT_DIR/'published_recovered_directions.pt',map_location='cpu',weights_only=True)
    fingerprints=[{'name':name,'rows':None,'v':v} for name,v in recovered.items()]
else:
    # The CPU worker may be queued; measured sparse forensics are sufficient to
    # motivate the single-setting causal test without allocating another job.
    AUDIT=RANGE_AUDIT
    fingerprints=[{'name':row['name'],'rows':row['sampled_output_rows'],
                   'v':torch.tensor(row['sampled_delta_dominant_direction'])} for row in DIRECTION_AUDITS]
replay_source={'id':'Observer04/Qwen3.5-9B-heretic-replay-20261107',
               'revision':'ff1b5ee362bf0817e2615e63fef01b3e2639c545',
               'filename':'experiments/refusal_directions.pt'}
means_path=hf_hub_download(replay_source['id'],replay_source['filename'],revision=replay_source['revision'])
prior_means=torch.load(means_path,map_location='cpu',weights_only=True)
old_directions=prior_means['directions'].float()
helper_path = OUT / 'evaluation.py'
helper_path.write_text(EVALUATION_SOURCE)
spec = importlib.util.spec_from_file_location('recovery_evaluation', helper_path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
manifest['recovery_audit_summary_sha256'] = hashlib.sha256(json.dumps(AUDIT,sort_keys=True).encode()).hexdigest()
manifest['evaluation_helper_sha256'] = hashlib.sha256(EVALUATION_SOURCE.encode()).hexdigest()
manifest['candidate_selection'] = 'Raw versus per-slot projected saved means; identical closed-thinking evaluation; dev-only selection'
manifest['direction_processing_note'] = ('effective_settings.orthogonalize_direction remains False because upstream main() '
    'is not called; per-candidate orthogonalization is explicitly applied to saved directions before Model.abliterate().')
manifest['reused_activation_means']={**replay_source,'sha256':hashlib.sha256(Path(means_path).read_bytes()).hexdigest(),
                                    'original_extraction':'400 harmful and400 harmless prompts, pinned first-phase run'}
manifest['historical_test_exposed'] = True
manifest['status'] = 'preparing_recovery_pilot'
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
(OUT/'weight_audit_summary.json').write_text(json.dumps(AUDIT,indent=2))
(OUT/'direction_fingerprints.json').write_text(json.dumps(DIRECTION_AUDITS,indent=2))

# Do not change evaluation formatting while comparing extraction prefixes.
native_template = model.tokenizer.apply_chat_template.func
def set_prefix(mode):
    assert mode in ('closed', 'open')
    model.tokenizer.apply_chat_template = partial(native_template, enable_thinking=(mode == 'open'))
    model.response_prefix = ''
set_prefix('closed')
def prompt_factory(text):
    return Prompt(system=settings.system_prompt, user=text)

data = {}
for label, dataset_id in [('good', 'mlabonne/harmless_alpaca'), ('bad', 'mlabonne/harmful_behaviors')]:
    data[label] = load_dataset(dataset_id, revision=PROVENANCE['datasets'][dataset_id])
hb = {}
for split in ('val', 'test'):
    url = ('https://raw.githubusercontent.com/centerforaisafety/HarmBench/' + ev.HARM_BENCH_REVISION
           + '/data/behavior_datasets/harmbench_behaviors_text_' + split + '.csv')
    with urllib.request.urlopen(url, timeout=45) as response:
        blob = response.read()
    hb[split] = ev.read_harmbench_csv(blob, split)
splits, split_manifest = ev.build_recovery_splits(
    list(data['good']['train']['text']), list(data['good']['test']['text']),
    list(data['bad']['train']['text']), list(data['bad']['test']['text']), hb['val'], hb['test'])
splits['dev_bad'] = splits['dev_bad_native'] + splits['dev_bad_external']
(OUT / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2))
with (OUT / 'prompts.jsonl').open('w') as handle:
    for label, records in splits.items():
        if label == 'dev_bad':
            continue
        for record in records:
            handle.write(json.dumps(dict(record, split=label, system=settings.system_prompt)) + '\n')
print('Recovery counts:', {k: len(v) for k, v in splits.items()}, flush=True)

def generate(records, condition, split, max_tokens=100):
    set_prefix('closed')
    return ev.generate_rows(model, records, prompt_factory, condition=condition, split=split,
                            batch_size=BATCH_SIZE, max_new_tokens=max_tokens,
                            output_path=OUT / f'{condition}_{split}_responses.jsonl')

def logprobs(records):
    set_prefix('closed')
    return ev.collect_logprobs(model, records, prompt_factory, batch_size=BATCH_SIZE)

baseline_dev_logprobs = logprobs(splits['dev_good'])
baseline_historical_logprobs = logprobs(splits['historical_good'])
baseline_fresh_logprobs = logprobs(splits['fresh_good'])
torch.save({'development': baseline_dev_logprobs, 'historical': baseline_historical_logprobs,
            'fresh': baseline_fresh_logprobs}, OUT / 'baseline_logprobs.pt')
baseline_dev_rows = generate(splits['dev_bad'], 'original', 'development')
baseline_dev = ev.summarize_rows(baseline_dev_rows)
(OUT / 'baseline_metrics.json').write_text(json.dumps(baseline_dev, indent=2))

@torch.inference_mode()
def extract(mode):
    set_prefix(mode)
    token_records = []
    means = {}
    for label in ('extract_good', 'extract_bad'):
        records = splits[label]
        prompts = [prompt_factory(r['prompt']) for r in records]
        total = None
        for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=mode+' '+label):
            states = model.get_residuals(prompts[start:start+BATCH_SIZE])
            total = states.sum(dim=0) if total is None else total + states.sum(dim=0)
        means[label] = total / len(prompts)
        for record, prompt in zip(records, prompts):
            rendered = model.tokenizer.apply_chat_template(
                [[{'role':'system','content':prompt.system},{'role':'user','content':prompt.user}]],
                add_generation_prompt=True, tokenize=False)[0]
            ids = model.tokenizer(rendered, return_token_type_ids=False)['input_ids']
            token_records.append({'id':record['id'], 'split':label,
                                  'rendered_sha256':ev.digest(rendered), 'token_count':len(ids),
                                  'last32_token_ids':ids[-32:]})
    directions = F.normalize(means['extract_bad'] - means['extract_good'], dim=1)
    assert directions.shape == (33,4096) and torch.isfinite(directions).all()
    torch.save({'good_means':means['extract_good'], 'bad_means':means['extract_bad'],
                'directions':directions}, OUT / f'{mode}_refusal_directions.pt')
    (OUT / f'{mode}_extraction_tokens.json').write_text(json.dumps(token_records))
    set_prefix('closed')
    return directions

good_directions=F.normalize(prior_means['good_means'].float(),p=2,dim=1)
projection=(old_directions*good_directions).sum(dim=1,keepdim=True)
projected_directions=F.normalize(old_directions-projection*good_directions,p=2,dim=1)
directions = {'raw':old_directions,'projected':projected_directions}
alignment = {}
for mode in ('raw', 'projected'):
    torch.save({'good_means':prior_means['good_means'],'bad_means':prior_means['bad_means'],
                'directions':directions[mode],'orthogonalize_direction':mode=='projected'},
                OUT/f'{mode}_refusal_directions.pt')
    idx = PUBLISHED['direction_index'] + 1
    selected = F.normalize(directions[mode][int(idx)].lerp(directions[mode][int(idx)+1],idx%1), dim=0)
    alignment[mode] = {'old_replay_slot_cosines':
                       (directions[mode]*old_directions).sum(dim=1).tolist(),
                       'publisher_delta_comparisons':[]}
    for fingerprint in fingerprints:
        name,v,rows=fingerprint['name'],fingerprint['v'],fingerprint['rows']
        compared=F.normalize(directions[mode][:,rows],dim=1) if rows is not None else directions[mode]
        compared_selected=F.normalize(selected[rows],dim=0) if rows is not None else selected
        similarities = torch.abs(compared @ v)
        alignment[mode]['publisher_delta_comparisons'].append({
            'name':name, 'sampled_output_rows':rows,'selected_abs_cosine':abs(torch.dot(compared_selected,v).item()),
            'best_slot':int(torch.argmax(similarities)), 'best_slot_abs_cosine':similarities.max().item()})
    (OUT / 'direction_alignment.json').write_text(json.dumps(alignment, indent=2))
    print('Direction processing', mode, 'selected cosine examples',
          [round(r['selected_abs_cosine'],5) for r in alignment[mode]['publisher_delta_comparisons'][:3]],flush=True)

# Candidates are frozen before any candidate development generations.
# Reuse our own pinned training means to change only the previously omitted flag.
# Per-slot projection+normalization occurs BEFORE interpolation, as upstream.
candidates = [{'candidate_id':'card_raw','extraction_mode':'raw','orthogonalize_direction':False},
              {'candidate_id':'card_projected','extraction_mode':'projected','orthogonalize_direction':True}]
(OUT / 'predeclared_candidates.json').write_text(json.dumps(candidates, indent=2))
parameters = {name:AbliterationParameters(**values) for name, values in PUBLISHED['parameters'].items()}
trials = []
for candidate in candidates:
    model.reset_model()
    set_prefix('closed')
    model.abliterate(directions[candidate['extraction_mode']],PUBLISHED['direction_index'],parameters)
    # Actual assigned PEFT tensors must implement the intended projection edit.
    checks = []
    for layer in (12,19,20,28):
        for component_name,modules in model.get_layer_modules(layer).items():
            p = PUBLISHED['parameters'][component_name]
            dist = abs(layer-p['max_weight_position'])
            lam = (p['max_weight'] + dist/p['min_weight_distance']*(p['min_weight']-p['max_weight'])) if dist<=p['min_weight_distance'] else 0.
            for module in modules:
                idx=PUBLISHED['direction_index']+1
                vectors=directions[candidate['extraction_mode']]
                v=F.normalize(vectors[int(idx)].lerp(vectors[int(idx)+1],idx%1),dim=0).to(module.weight.device)
                W=module.base_layer.weight.float()
                # Check tensor assignment and actual scaling without materializing a full delta matrix.
                expected_A=v@W
                actual_A=module.lora_A['default'].weight.float().view(-1)
                actual_B=module.lora_B['default'].weight.float().view(-1)
                checks.append({'layer':layer,'component':component_name,
                               'scaling':module.scaling['default'],
                               'A_max_error':(actual_A-expected_A).abs().max().item(),
                               'B_max_error':(actual_B+lam*v).abs().max().item()})
                assert module.scaling['default']==1
                assert checks[-1]['A_max_error']<1e-5 and checks[-1]['B_max_error']<1e-5, checks[-1]
                del W, expected_A, actual_A, actual_B, v
    (OUT / f"{candidate['candidate_id']}_adapter_checks.json").write_text(json.dumps(checks,indent=2))
    rows=generate(splits['dev_bad'],candidate['candidate_id'],'development')
    metrics=ev.summarize_rows(rows)
    kl=ev.first_token_kl(logprobs(splits['dev_good']),baseline_dev_logprobs)
    trial=dict(candidate,selection_split='development',**metrics,benign_first_token_kl=kl)
    trials.append(trial)
    (OUT / 'development_trials.json').write_text(json.dumps(trials,indent=2))
    print('DEVELOPMENT',json.dumps(trial),flush=True)

winner=ev.choose_development_candidate(trials)
(OUT / 'frozen_selection.json').write_text(json.dumps(winner,indent=2))

def validate_selected():
    model.reset_model()
    model.abliterate(directions[winner['extraction_mode']],PUBLISHED['direction_index'],parameters)
    set_prefix('closed')
    model.model.save_pretrained(OUT / 'selected_adapter',safe_serialization=True)
    model.tokenizer.save_pretrained(OUT / 'selected_adapter')
    adapter_config_path=OUT / 'selected_adapter/adapter_config.json'
    adapter_config=json.loads(adapter_config_path.read_text())
    adapter_config['base_model_name_or_path']=PROVENANCE['base_model']['id']
    adapter_config['revision']=PROVENANCE['base_model']['revision']
    adapter_config_path.write_text(json.dumps(adapter_config,indent=2))
    summary={'baseline_development':baseline_dev,'trials':trials,'frozen_selection':winner,
             'scope':'Single-setting raw/projected causal pilot; historical marker/KL comparability, not full HarmBench',
             'semantic_harmful_compliance':'not automatically classified'}
    # Avoid consuming new test data/long generations for an obviously weak pilot.
    # This is a runtime-saving rule, not a claim of published-level success.
    if winner['marker_refusal_rate']>=.5 or winner['benign_first_token_kl']>=.3:
        summary['continuation']='Stopped after development; repair remains unresolved.'
        (OUT / 'summary.json').write_text(json.dumps(summary,indent=2))
        manifest['status']='pilot_stopped_after_development'
        (OUT / 'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        print('RECOVERY PILOT STOPPED',json.dumps(summary),flush=True)
        return
    unmerged_path=OUT / f"{winner['candidate_id']}_development_responses.jsonl"
    unmerged_rows=[json.loads(line) for line in unmerged_path.read_text().splitlines()]
    unmerged_logprobs=logprobs(splits['dev_good'])
    model.model=model.get_merged_model()
    model.model.eval()
    merged_dev=generate(splits['dev_bad'],'selected_merged','development')
    merged_logprobs=logprobs(splits['dev_good'])
    summary['merge_comparison']={
        'n':len(merged_dev),
        'exact_response_matches':sum(a['response']==b['response'] for a,b in zip(unmerged_rows,merged_dev)),
        'unmerged_development_kl':ev.first_token_kl(unmerged_logprobs,baseline_dev_logprobs),
        'merged_development_kl':ev.first_token_kl(merged_logprobs,baseline_dev_logprobs),
        'merged_to_unmerged_first_token_kl':ev.first_token_kl(merged_logprobs,unmerged_logprobs),
        'max_abs_logprob_difference':(merged_logprobs-unmerged_logprobs).abs().max().item()}
    historical_rows=generate(splits['historical_bad'],'selected_merged','historical_exposed')
    historical_metrics=ev.summarize_rows(historical_rows)
    historical_metrics['benign_first_token_kl']=ev.first_token_kl(logprobs(splits['historical_good']),baseline_historical_logprobs)
    fresh_rows=generate(splits['fresh_bad'],'selected_merged','fresh_standard_subset')
    fresh_metrics=ev.summarize_rows(fresh_rows)
    fresh_metrics['benign_first_token_kl']=ev.first_token_kl(logprobs(splits['fresh_good']),baseline_fresh_logprobs)
    long_rows=generate(splits['fresh_bad'][:12],'selected_merged','fresh_long_subset',512)
    summary.update({'selected_merged_development':ev.summarize_rows(merged_dev),
                    'selected_historical_exposed':historical_metrics,'selected_fresh_subset':fresh_metrics,
                    'selected_long_subset':ev.summarize_rows(long_rows)})
    (OUT / 'summary.json').write_text(json.dumps(summary,indent=2))
    queue,mapping=ev.make_blind_review_queue(historical_rows+fresh_rows+long_rows)
    for name,values in [('blind_review_queue',queue),('blind_review_mapping',mapping)]:
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    long_queue,long_mapping=ev.make_blind_review_queue(long_rows,n_prompts=12)
    for name,values in [('long_only_blind_review_queue',long_queue),('long_only_blind_review_mapping',long_mapping)]:
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    summary['review_scope']='Single-condition diagnostic queue; paired original/publisher validation deferred.'
    (OUT / 'summary.json').write_text(json.dumps(summary,indent=2))
    manifest['status']='pilot_complete'
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('RECOVERY PILOT COMPLETE',json.dumps(summary),flush=True)

validate_selected()
