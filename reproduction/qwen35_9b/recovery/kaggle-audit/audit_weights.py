"""CPU-only forensic audit. No model is instantiated and no generation is run."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

assert Path('/kaggle').exists(), 'Run on Kaggle, not locally.'
START = time.time()
OUT = Path('/kaggle/working/recovery_audit')
OUT.mkdir(parents=True, exist_ok=True)
for credential in Path('/kaggle/input').glob('**/phase1_hf_token.txt'):
    os.environ['HF_TOKEN'] = credential.read_text().strip()
    break
os.environ['HF_HOME'] = '/kaggle/temp/recovery_audit_hf'
Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)
free_gib = shutil.disk_usage(os.environ['HF_HOME']).free / 2**30
print('CPU audit free temp GiB:', round(free_gib, 2), flush=True)
assert free_gib > 37, 'Need space for original and published shards; no GPU job is needed.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'huggingface-hub==1.33.0', 'safetensors==0.8.0'], check=True)
import torch
from safetensors import safe_open
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

torch.set_grad_enabled(False)
torch.set_num_threads(4)
assert not torch.cuda.is_available(), 'This diagnostic must use CPU quota only.'
api = HfApi()
BASE = {'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'}
PUB = {'id': 'trohrbaugh/Qwen3.5-9B-heretic-v2', 'revision': '81dda62421c43a5bd5722edbc6cd47714673dc7f'}
REPLAY = {'id': 'Observer04/Qwen3.5-9B-heretic-replay-20261107',
          'revision': 'ff1b5ee362bf0817e2615e63fef01b3e2639c545'}
PARAMS = {'direction_index': 18.49, 'parameters': {
    'attn.o_proj': {'max_weight': 2.0, 'max_weight_position': 18.99,
                    'min_weight': 1.43, 'min_weight_distance': 9.65},
    'mlp.down_proj': {'max_weight': 1.67, 'max_weight_position': 26.06,
                    'min_weight': .84, 'min_weight_distance': 18.04}}}
manifest = {'base': BASE, 'published': PUB, 'replay': REPLAY, 'published_parameters': PARAMS,
            'torch': torch.__version__, 'mode': 'CPU tensor-file forensic audit; no model execution',
            'sample_limit': 4096, 'cpu_threads': torch.get_num_threads()}
(OUT / 'audit_manifest.json').write_text(json.dumps(manifest, indent=2))

paths = {}
indexes = {}
handles = {}
for label, source in [('base', BASE), ('published', PUB)]:
    print('Downloading pinned tensor files for', label, flush=True)
    paths[label] = Path(snapshot_download(source['id'], revision=source['revision'], max_workers=4,
                         allow_patterns=['*.safetensors', 'model.safetensors.index.json']))
    indexes[label] = json.loads((paths[label] / 'model.safetensors.index.json').read_text())['weight_map']
    handles[label] = {shard: safe_open(str(paths[label] / shard), framework='pt', device='cpu')
                      for shard in sorted(set(indexes[label].values()))}

def read(label, name, full=False):
    handle = handles[label][indexes[label][name]]
    view = handle.get_slice(name)
    if full or len(view.get_shape()) <= 1:
        return handle.get_tensor(name)
    shape = view.get_shape()
    # Read first/last rows rather than materializing huge embedding matrices.
    rows = min(shape[0], max(1, 4096 // max(1, shape[-1])))
    a = view[:rows].reshape(-1)[:4096]
    b = view[-rows:].reshape(-1)[-4096:]
    return torch.cat([a, b])

common = sorted(set(indexes['base']) & set(indexes['published']))
summary = {'common_tensors': len(common),
           'base_only': sorted(set(indexes['base']) - set(indexes['published'])),
           'published_only': sorted(set(indexes['published']) - set(indexes['base'])),
           'sample_changed': [], 'sample_unchanged_count': 0, 'projection_stats': []}

def component(name):
    if '.mlp.down_proj.weight' in name:
        return 'mlp.down_proj'
    if '.self_attn.o_proj.weight' in name or '.linear_attn.out_proj.weight' in name:
        return 'attn.o_proj'
    return None

with (OUT / 'sample_tensor_comparison.jsonl').open('w') as f:
    for name in common:
        a, b = read('base', name), read('published', name)
        # Published storage is BF16 throughout, including formerly FP32 buffers.
        # Compare against the original after conversion to the published dtype.
        cast = a.to(b.dtype)
        d = b.float() - cast.float()
        changed = int(torch.count_nonzero(d))
        row = {'name': name, 'base_dtype': str(a.dtype), 'published_dtype': str(b.dtype),
               'sample_n': d.numel(), 'changed_after_dtype_match': changed,
               'max_abs_delta': d.abs().max().item(), 'component': component(name)}
        f.write(json.dumps(row) + '\n')
        if changed:
            summary['sample_changed'].append(row)
        else:
            summary['sample_unchanged_count'] += 1
        del a, b, cast, d
print('Sparse audit:', len(summary['sample_changed']), 'changed tensors of', len(common), flush=True)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

# The previous run's small direction/adapter evidence is enough; do not download its19GB merged weights.
directions_path = hf_hub_download(REPLAY['id'], 'experiments/refusal_directions.pt', revision=REPLAY['revision'])
replay_state = torch.load(directions_path, map_location='cpu', weights_only=True)
replay_vectors = replay_state['directions'].float()
index = PARAMS['direction_index'] + 1
replay_selected = torch.nn.functional.normalize(
    torch.lerp(replay_vectors[int(index)], replay_vectors[int(index)+1], index % 1), dim=0)
torch.save({'directions': replay_vectors, 'selected': replay_selected}, OUT / 'replay_directions.pt')
recovered = {}
generator = torch.Generator().manual_seed(4939)

for name in common:
    comp = component(name)
    if comp is None or '.language_model.layers.' not in name:
        continue
    layer = int(name.split('.layers.')[1].split('.')[0])
    a, b = read('base', name, True), read('published', name, True)
    W = a.to(b.dtype).float()
    D = b.float() - W
    norm2 = D.square().sum().item()
    params = PARAMS['parameters'][comp]
    dist = abs(layer - params['max_weight_position'])
    expected_weight = (params['max_weight'] + dist / params['min_weight_distance'] *
                      (params['min_weight'] - params['max_weight'])) if dist <= params['min_weight_distance'] else 0.
    row = {'name': name, 'layer': layer, 'component': comp, 'shape': list(D.shape),
           'delta_frobenius': norm2 ** .5, 'expected_weight_from_rounded_card': expected_weight,
           'changed_after_dtype_match': int(torch.count_nonzero(D))}
    if norm2 > 0:
        # For deltaW=-lambda*v*(v^T W), D times any nonorthogonal random vector
        # recovers v. A short subspace iteration makes this robust to BF16 rounding.
        x = torch.randn((D.shape[1], 2), generator=generator)
        sketch = D @ x
        q = torch.linalg.qr(sketch, mode='reduced').Q
        for _ in range(2):
            q = torch.linalg.qr(D @ (D.T @ q), mode='reduced').Q
        small = q.T @ D
        U, S, _ = torch.linalg.svd(small, full_matrices=False)
        v = torch.nn.functional.normalize(q @ U[:, 0], dim=0)
        if v[torch.argmax(v.abs())] < 0:
            v = -v
        direction_row = v @ D
        base_row = v @ W
        inferred_weight = -torch.dot(direction_row, base_row) / base_row.square().sum()
        retained = direction_row.square().sum().item() / norm2
        cosines = torch.abs(replay_vectors @ v)
        row.update({'dominant_rank1_energy_fraction': retained,
                    'inferred_ablation_weight': inferred_weight.item(),
                    'abs_cosine_with_replay_selected': abs(torch.dot(v, replay_selected).item()),
                    'best_replay_hidden_slot': int(torch.argmax(cosines)),
                    'best_replay_hidden_slot_abs_cosine': cosines.max().item()})
        recovered[name] = v
        # Verify matching output intervention equals analytical weight edit in FP32;
        # separately quantify the rounding error incurred by FP16 merge/storage.
        if expected_weight:
            inputs = torch.randn((W.shape[1], 3), generator=generator)
            base_fp16 = a.half().float()
            analytical_delta = -expected_weight * torch.outer(replay_selected, replay_selected @ base_fp16)
            baseline_output = base_fp16 @ inputs
            intervention_output = baseline_output - expected_weight * replay_selected[:, None] * (
                replay_selected @ baseline_output)[None, :]
            adapter_output = baseline_output + analytical_delta @ inputs
            merged_output = (base_fp16 + analytical_delta).half().float() @ inputs
            scale = torch.linalg.vector_norm(intervention_output)
            row['projection_intervention_adapter_relative_error_fp32'] = (
                torch.linalg.vector_norm(intervention_output - adapter_output) / scale).item()
            row['projection_intervention_fp16_merge_relative_error'] = (
                torch.linalg.vector_norm(intervention_output - merged_output) / scale).item()
            del inputs, base_fp16, analytical_delta, baseline_output, intervention_output, adapter_output, merged_output
        del x, sketch, q, small, U, S, v, direction_row, base_row, cosines
    summary['projection_stats'].append(row)
    print('Projection', layer, name.split('.layers.')[1], 'delta', round(row['delta_frobenius'], 5),
          'cos', round(row.get('abs_cosine_with_replay_selected', 0.), 5), flush=True)
    del a, b, W, D
    gc.collect()
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

torch.save(recovered, OUT / 'published_recovered_directions.pt')
keys = list(recovered)
summary['direction_consistency'] = [
    {'reference': keys[0], 'name': k, 'abs_cosine': abs(torch.dot(recovered[keys[0]], recovered[k]).item())}
    for k in keys] if keys else []
summary['elapsed_seconds'] = time.time() - START
summary['complete'] = True
summary['limits'] = ['Non-target tensors were sampled, not exhaustively compared.',
                     'Recovered directions are inferred from rounded published weight deltas.',
                     'Projection math checks are analytical CPU checks, not full-model hook tests.']
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print('AUDIT COMPLETE', json.dumps({'changed_samples': len(summary['sample_changed']),
      'edited_projections': len(recovered), 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
