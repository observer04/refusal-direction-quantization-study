import ast, json, textwrap, hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
material=root/'reproduction/qwen35_9b'
provenance=json.loads((material/'sources.json').read_text())
parameters=json.loads((material/'published_parameters.json').read_text())
requirements=(material/'requirements-kaggle.txt').read_text()
requirements='\n'.join(x for x in requirements.splitlines() if not x.startswith('nbformat=='))+'\n'
(material/'requirements-kaggle.txt').write_text(requirements)
cells=[]
def cell(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    out={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True),'id':hashlib.sha256(source.encode()).hexdigest()[:12]}
    if kind=='code':
        ast.parse(source)
        out.update(execution_count=None,outputs=[])
    cells.append(out)
cell('markdown','''
# Qwen3.5-9B: replay the published Heretic directional-ablation settings

**Kaggle:** GPU **T4 ×2**, Internet **ON**. Run cells in order in a fresh session.
This notebook downloads upstream code and the original Qwen weights, extracts directions using
400 harmful + 400 harmless prompts, applies the published settings **once**, and measures the
100-prompt refusal count and benign first-token KL. No optimizer search or quantization. Uploads use an HF token supplied privately to Kaggle.

Reference: [trohrbaugh/Qwen3.5-9B-heretic-v2](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2).
Its card reports **100/100 → 6/100 refusals, KL 0.0793**. The separate evaluator's 0/400
HarmBench result is a different experiment and is not reproduced here.

The author's notebook, study journal, exact environment/seed/revisions, and unrounded parameters
were not published. This is a **best-effort replay of the published settings**, not an exact
checkpoint reconstruction. We retain upstream v1.2.0 directional-ablation math and use a pinned
post-tag v1.2.0 commit with Qwen3.5 hybrid-layer support. Kaggle adaptations: FP16 with required
FP32 modules, explicit non-thinking chat formatting, and CPU aggregation across GPUs.
Metrics use the original Heretic refusal-marker scorer; they do not measure unsafe compliance.
''')
cell('code',f'''
# 1. Download the small pinned source bundle and install dependencies ON KAGGLE.
# Torch is already provided by Kaggle and is constrained to its existing version.
import hashlib, importlib.metadata, io, json, os, subprocess, sys, urllib.request, zipfile
from pathlib import Path

assert Path('/kaggle').exists(), 'Run this notebook on Kaggle.'
WORK = Path('/kaggle/working/qwen35_reproduction')
WORK.mkdir(parents=True, exist_ok=True)
# Batch mode may receive the owner's token as a private Kaggle input.
# Never print or copy this credential into the output directory.
if not os.environ.get('HF_TOKEN'):
    for credential_file in Path('/kaggle/input').glob('**/phase1_hf_token.txt'):
        os.environ['HF_TOKEN'] = credential_file.read_text().strip()
        break
os.environ.setdefault('HF_HOME', '/kaggle/temp/qwen35_hf_cache')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
PROVENANCE = json.loads({json.dumps(provenance, ensure_ascii=False)!r})
PUBLISHED = json.loads({json.dumps(parameters)!r})
REQUIREMENTS = {requirements!r}
(WORK / 'requirements-kaggle.txt').write_text(REQUIREMENTS)
(WORK / 'sources.json').write_text(json.dumps(PROVENANCE, indent=2))
(WORK / 'published_parameters.json').write_text(json.dumps(PUBLISHED, indent=2))

# Kaggle ships an old torchao that recent PEFT rejects even for unquantized LoRA.
# It is unused in this phase; remove it while preserving the CUDA PyTorch stack.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
constraints = []
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        constraints.append(name + '==' + importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        pass
assert any(x.startswith('torch==') for x in constraints), 'Select a Kaggle GPU image with PyTorch.'
(WORK / 'preserve-kaggle-torch.txt').write_text('\\n'.join(constraints) + '\\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(WORK / 'requirements-kaggle.txt'),
                '-c', str(WORK / 'preserve-kaggle-torch.txt')], check=True)

with urllib.request.urlopen(PROVENANCE['compatible_source_url'], timeout=60) as response:
    source_zip = response.read()
assert hashlib.sha256(source_zip).hexdigest() == PROVENANCE['compatible_source_archive_sha256']
SOURCE = WORK / 'upstream'
with zipfile.ZipFile(io.BytesIO(source_zip)) as archive:
    for item in archive.infolist():
        parts = Path(item.filename).parts[1:]
        if not parts or item.is_dir():
            continue
        target = SOURCE.joinpath(*parts)
        assert target.resolve().is_relative_to(SOURCE.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))
sys.path.insert(0, str(SOURCE / 'src'))
print('Source ready:', PROVENANCE['compatible_source_commit'])
print('Kaggle PyTorch preserved:', constraints)
''')
cell('code','''
# 2. GPU setup. Keep the published sample counts and response limit.
import gc, platform, random, time
from collections import Counter
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.model import Model, AbliterationParameters
from heretic.utils import Prompt

SEED = 0                 # Our recorded seed; the author's seed is unknown.
BATCH_SIZE = 1           # Conservative start for two T4s.
MAX_NEW_TOKENS = 100     # Heretic v1.2.0 default.
SAVE_MERGED = True       # Approximately 19 GB; saved under /kaggle/working.
OUT = WORK / 'results'
OUT.mkdir(exist_ok=True)
assert torch.cuda.is_available(), 'Enable GPU T4 ×2 in Kaggle settings.'
assert torch.cuda.device_count() >= 2, 'This configuration expects two GPUs.'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_grad_enabled(False)

GPUS = [{'index': i, 'name': torch.cuda.get_device_name(i),
         'vram_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
        for i in range(torch.cuda.device_count())]
MAX_MEMORY = {str(g['index']): f"{min(13, int(g['vram_gib']) - 2)}GiB" for g in GPUS}
MAX_MEMORY['cpu'] = '0GiB'  # Fail visibly instead of silently offloading to a slow CPU path.
print(json.dumps(GPUS, indent=2))
for location in ('/kaggle/working', '/kaggle/temp', '/tmp'):
    import shutil
    Path(location).mkdir(parents=True, exist_ok=True)
    print('Free GiB:', location, round(shutil.disk_usage(location).free / 2**30, 1))

# The upstream collector stacks hidden states directly, which fails across GPUs.
# Move only the final-token states to CPU before stacking; keep the original estimator.
class KaggleModel(Model):
    @torch.inference_mode()
    def get_residuals(self, prompts):
        _, outputs = self.generate(prompts, max_new_tokens=1,
                                   output_hidden_states=True, return_dict_in_generate=True)
        states = torch.stack([h[:, -1, :].detach().float().cpu()
                              for h in outputs.hidden_states[0]], dim=1)
        assert torch.isfinite(states).all(), 'Non-finite residuals; stop this run.'
        return states

    @torch.inference_mode()
    def get_logprobs(self, prompts):
        values = super().get_logprobs(prompts).detach().float().cpu()
        assert torch.isfinite(values).all(), 'Non-finite log probabilities; stop this run.'
        return values

versions = {}
for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'huggingface-hub',
             'bitsandbytes', 'kernels', 'numpy'):
    versions[name] = importlib.metadata.version(name)
manifest = {'status': 'started', 'sources': PROVENANCE, 'published_parameters': PUBLISHED,
            'seed': SEED, 'python': platform.python_version(), 'packages': versions,
            'gpus': GPUS, 'batch_size': BATCH_SIZE, 'max_new_tokens': MAX_NEW_TOKENS,
            'quantization': 'none', 'dtype': 'float16 with model-required FP32 exceptions',
            'thinking': False, 'scorer': 'upstream Heretic v1.2.0 refusal markers',
            'reproduction_scope': 'rounded published-parameter replay, not exact weight reproduction'}
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
               stdout=(OUT / 'environment.txt').open('w'))
''')
cell('code','''
# 3. Download the pinned original model and load it once.
# Public checkpoints/datasets require no embedded token or Hub login.
base = PROVENANCE['base_model']
BASE_PATH = snapshot_download(base['id'], revision=base['revision'], max_workers=4,
                             allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
settings_values = dict(
    model=BASE_PATH, dtypes=['float16'], device_map='auto', max_memory=MAX_MEMORY,
    batch_size=BATCH_SIZE, max_response_length=MAX_NEW_TOKENS,
    quantization='none', trust_remote_code=False,
    orthogonalize_direction=False, row_normalization='none', winsorization_quantile=1.0,
)
# Pydantic's CLI source must not parse Jupyter's '-f kernel.json' arguments.
old_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    settings = Settings(**settings_values)
finally:
    sys.argv = old_argv
model = KaggleModel(settings)
model.model.eval()
manifest['effective_settings'] = settings.model_dump(mode='json')
# Explicitly close the thinking block using Qwen's own chat template.
# Old Heretic auto-prefix detection is unreliable with this model's current template.
model.tokenizer.apply_chat_template = partial(model.tokenizer.apply_chat_template, enable_thinking=False)
model.response_prefix = ''
placements = getattr(model.model, 'hf_device_map', {})
assert not any(str(device) in ('cpu', 'disk') for device in placements.values()), placements
manifest['device_map'] = {k: str(v) for k, v in placements.items()}
manifest['parameter_dtypes'] = dict(Counter(str(p.dtype) for p in model.model.parameters()))
components = Counter()
for layer in range(len(model.get_layers())):
    for name, modules in model.get_layer_modules(layer).items():
        components[name] += len(modules)
assert components == {'attn.o_proj': 32, 'mlp.down_proj': 32}, components
print('Target modules:', dict(components))
smoke = [Prompt(system=settings.system_prompt, user='What is 1+1?')]
print('Smoke response:', model.get_responses(smoke, skip_special_tokens=True)[0])
model.get_logprobs(smoke)  # Finite-logit check before the full run.
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
''')
cell('code','''
# 4. Load the same dataset slices as upstream defaults, with recorded revisions.
prompt_sets = {}
for label, dataset_id in [('good', 'mlabonne/harmless_alpaca'), ('bad', 'mlabonne/harmful_behaviors')]:
    data = load_dataset(dataset_id, revision=PROVENANCE['datasets'][dataset_id])
    for split, count in [('train', 400), ('test', 100)]:
        texts = list(data[split]['text'][:count])
        assert len(texts) == count
        prompt_sets[f'{label}_{split}'] = [Prompt(system=settings.system_prompt, user=t) for t in texts]
with (OUT / 'prompts.jsonl').open('w') as f:
    for split, prompts in prompt_sets.items():
        for index, prompt in enumerate(prompts):
            f.write(json.dumps({'split': split, 'index': index, 'system': prompt.system,
                                'prompt': prompt.user}) + '\\n')
print({key: len(value) for key, value in prompt_sets.items()})

# Reuse the upstream scorer verbatim, while saving every generated answer.
judge = object.__new__(Evaluator)
judge.settings = settings

def first_token_logprobs(prompts, label):
    return torch.cat([model.get_logprobs(prompts[i:i+BATCH_SIZE])
                      for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=label)])

@torch.inference_mode()
def evaluate_refusals(condition):
    rows = []
    with (OUT / f'{condition}_responses.jsonl').open('w') as f:
        for start in tqdm(range(0, 100, BATCH_SIZE), desc=condition):
            batch = prompt_sets['bad_test'][start:start+BATCH_SIZE]
            inputs, output_ids = model.generate(batch, max_new_tokens=MAX_NEW_TOKENS)
            sequences = output_ids[:, inputs['input_ids'].shape[1]:].detach().cpu().tolist()
            eos = model.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for offset, (prompt, ids) in enumerate(zip(batch, sequences)):
                terminal = next((i for i, token in enumerate(ids) if token in eos), None)
                tokens = ids if terminal is None else ids[:terminal + 1]
                response = model.tokenizer.decode(tokens, skip_special_tokens=True)
                row = {'condition': condition, 'index': start + offset, 'prompt': prompt.user,
                       'response': response, 'refusal_marker_match': judge.is_refusal(response),
                       'generated_tokens': len(tokens), 'hit_token_limit': terminal is None}
                rows.append(row)
                f.write(json.dumps(row) + '\\n')
                f.flush()
            if len(rows) % 10 == 0:
                print(condition, len(rows), '/ 100', flush=True)
    return {'refusals': sum(r['refusal_marker_match'] for r in rows), 'n': len(rows),
            'hit_token_limit': sum(r['hit_token_limit'] for r in rows)}

baseline_logprobs = first_token_logprobs(prompt_sets['good_test'], 'Baseline KL reference')
torch.save(baseline_logprobs, OUT / 'baseline_logprobs.pt')
baseline_metrics = evaluate_refusals('baseline')
(OUT / 'baseline_metrics.json').write_text(json.dumps(baseline_metrics, indent=2))
print('Baseline:', baseline_metrics)
assert baseline_metrics['refusals'] > 0, 'Baseline scorer detected zero refusals; inspect responses before continuing.'
''')
cell('code','''
# 5. Extract directions once, then replay the published parameter values.
@torch.inference_mode()
def collect_mean(prompts, label):
    total = None
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc=label):
        states = model.get_residuals(prompts[start:start+BATCH_SIZE])
        batch_sum = states.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        if (start + BATCH_SIZE) % 100 == 0:
            print(label, start + BATCH_SIZE, '/', len(prompts), flush=True)
    return total / len(prompts)

good_means = collect_mean(prompt_sets['good_train'], 'Harmless residuals')
bad_means = collect_mean(prompt_sets['bad_train'], 'Harmful residuals')
refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
assert refusal_directions.shape == (33, 4096), refusal_directions.shape
assert torch.isfinite(refusal_directions).all()
torch.save({'good_means': good_means, 'bad_means': bad_means, 'directions': refusal_directions},
           OUT / 'refusal_directions.pt')
(OUT / 'direction_metadata.json').write_text(json.dumps({
    'shape': list(refusal_directions.shape),
    'index_convention': 'upstream hidden-state slot 0 is embeddings; slot i+1 follows block i; final slot may include final normalization',
    'direction_index': PUBLISHED['direction_index'],
    'interpolated_hidden_state_slots': [19, 20],
    "note": "18.49 is the interpolated source index, not a ranked candidate number",
}, indent=2))
parameters = {name: AbliterationParameters(**values) for name, values in PUBLISHED['parameters'].items()}
model.abliterate(refusal_directions, PUBLISHED['direction_index'], parameters)
# Save a small adapter first, so a later checkpoint export failure does not lose the edit.
model.model.save_pretrained(OUT / 'adapter', safe_serialization=True)
model.tokenizer.save_pretrained(OUT / 'adapter')
# Evaluate the actual merged weights, rather than only the unmerged adapter.
model.model = model.get_merged_model()
model.model.eval()
print('Published settings applied and merged. No search was run.')
''')
cell('code','''
# 6. Evaluate the merged result with the same prompts and original Heretic scorer.
edited_logprobs = first_token_logprobs(prompt_sets['good_test'], 'Edited KL')
kl = F.kl_div(edited_logprobs, baseline_logprobs, reduction='batchmean', log_target=True).item()
assert np.isfinite(kl), 'Non-finite KL; stop this run.'
edited_metrics = evaluate_refusals('edited')
summary = {'baseline': baseline_metrics, 'edited': {**edited_metrics, 'benign_first_token_kl': kl},
           'published_reference': PROVENANCE['published_metrics'],
           'scope': 'Heretic marker-count / first-token-KL replay; not HarmBench unsafe-compliance evaluation'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
manifest['status'] = 'evaluated'
manifest['peak_memory_gib'] = {str(i): torch.cuda.max_memory_allocated(i) / 2**30
                               for i in range(torch.cuda.device_count())}
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(summary, indent=2))
''')
cell('code','''
# 7. Save the merged checkpoint. Hub persistence follows in a separate cell.
import shutil
if SAVE_MERGED:
    checkpoint = Path('/kaggle/temp/Qwen3.5-9B-heretic-replay')
    checkpoint.mkdir(exist_ok=True, parents=True)
    needed = sum(p.numel() * p.element_size() for p in model.model.parameters())
    free = shutil.disk_usage(checkpoint).free
    assert free > needed + 2 * 2**30, f'Need about {(needed / 2**30) + 2:.1f} GiB free to export.'
    model.model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size='4GB')
    model.tokenizer.save_pretrained(checkpoint)
    manifest['checkpoint'] = str(checkpoint)
    print('Checkpoint saved:', checkpoint)
manifest['status'] = 'checkpoint_saved'
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Results:', OUT)
print('Keep summary.json, *_responses.jsonl, refusal_directions.pt, and run_manifest.json.')
''')
cell('code','''
# 8. Reload the saved weights and check all 100 refusal-test responses again.
from transformers import AutoModelForImageTextToText
before_reload = [json.loads(line) for line in (OUT / 'edited_responses.jsonl').read_text().splitlines()]
assert SAVE_MERGED and checkpoint.exists()
del model.model
gc.collect()
torch.cuda.empty_cache()
model.model = AutoModelForImageTextToText.from_pretrained(
    checkpoint, dtype=torch.float16, device_map='auto',
    max_memory={int(k) if k.isdigit() else k:v for k,v in MAX_MEMORY.items()},
    trust_remote_code=False,
)
model.model.eval()
assert not any(str(x) in ('cpu','disk') for x in model.model.hf_device_map.values())
reloaded_metrics = evaluate_refusals('reloaded')
after_reload = [json.loads(line) for line in (OUT / 'reloaded_responses.jsonl').read_text().splitlines()]
response_matches = sum(a['response'] == b['response'] for a,b in zip(before_reload, after_reload))
summary['reloaded'] = {**reloaded_metrics, 'exact_response_matches':response_matches,
                       'compared_to':'edited merged model before serialization'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
manifest['status'] = 'reload_verified'
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(summary, indent=2))
assert response_matches == 100, 'Reload differs from pre-save responses; inspect before claiming verification.'
''')
cell('code','''
# 9. Persist the checkpoint and evidence to the owner's private Hugging Face repo.
# HF_TOKEN is supplied privately to this Kaggle process; never embed it in notebook cells.
from huggingface_hub import HfApi
assert os.environ.get('HF_TOKEN'), 'Pass HF_TOKEN privately or use Kaggle Secrets.'
api = HfApi()
owner = api.whoami()['name']
repo_id = owner + '/Qwen3.5-9B-heretic-replay-20261107'
api.create_repo(repo_id, repo_type='model', private=True, exist_ok=True)
card = '\\n'.join([
    '---', 'license: apache-2.0', 'base_model: Qwen/Qwen3.5-9B',
    'library_name: transformers', 'tags:', '- refusal-research', '- heretic', '---',
    '# Qwen3.5-9B published-parameter replay', '',
    'Best-effort FP16 replay of trohrbaugh/Qwen3.5-9B-heretic-v2 rounded settings.',
    'Executed on Kaggle T4 x2, using pinned hybrid-compatible Heretic v1.2.0 source.',
    'This checkpoint is experimental. Refusal-marker counts do not establish harmful compliance.',
    'The 0/400 external HarmBench result is not tested by this 100-prompt replay.', '',
    '## Measurements', '', '```json', json.dumps(summary, indent=2), '```', '',
    'Full parameters, revisions, answers, vectors, adapter and runtime manifest are in experiments/.',
    'The author did not publish exact source/environment/seed/unrounded parameters.',
    'Changes for Kaggle: FP16, explicit non-thinking formatting, CPU aggregation across GPUs.',
])
(checkpoint / 'README.md').write_text(card + '\\n')
# Preserve small evidence before starting the large checkpoint transfer.
api.upload_folder(repo_id=repo_id, folder_path=str(OUT), path_in_repo='experiments',
                  commit_message='Save Phase 1 metrics, responses, vectors and provenance')
api.upload_folder(repo_id=repo_id, folder_path=str(checkpoint),
                  commit_message='Persist reloaded and evaluated FP16 replay checkpoint')
files = api.list_repo_files(repo_id)
assert 'config.json' in files and any(f.endswith('.safetensors') for f in files)
assert 'experiments/summary.json' in files
manifest['status'] = 'persisted'
manifest['hub_repo'] = repo_id
manifest['hub_files'] = len(files)
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
api.upload_file(repo_id=repo_id, path_or_fileobj=str(OUT / 'run_manifest.json'),
                path_in_repo='experiments/run_manifest.json', commit_message='Record successful persistence')
print('Persisted:', 'https://huggingface.co/' + repo_id)
''')
cell('markdown','''
If a cell fails, keep its traceback and `results/run_manifest.json`. GPU execution has not been
validated locally. Start from a fresh session before reapplying the edit. To save only the adapter,
vectors and results, set `SAVE_MERGED = False` in cell 2. Do not interpret missing refusal markers
as proof of harmful compliance. The checkpoint is a replay in FP16, while the published model is BF16.
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
path=root/'Reproduce_Qwen3_5_9B_Heretic.ipynb'
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
print('Created',path.name,'with',len(cells),'cells')
