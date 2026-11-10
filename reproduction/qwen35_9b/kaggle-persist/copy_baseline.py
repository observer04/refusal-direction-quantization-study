"""CPU-only transfer of the evaluated upstream baseline; no model execution."""
import json,os,sys,subprocess,shutil
from pathlib import Path
assert Path('/kaggle').exists()
for p in Path('/kaggle/input').glob('**/phase1_hf_token.txt'):
    os.environ['HF_TOKEN']=p.read_text().strip()
    break
assert os.environ.get('HF_TOKEN')
os.environ['HF_HOME']='/kaggle/temp/hf_baseline_transfer'
subprocess.run([sys.executable,'-m','pip','install','-q','huggingface-hub==1.33.0'],check=True)
from huggingface_hub import HfApi,snapshot_download
api=HfApi()
SRC='trohrbaugh/Qwen3.5-9B-heretic-v2'
REV='81dda62421c43a5bd5722edbc6cd47714673dc7f'
DEST=api.whoami()['name']+'/Qwen3.5-9B-heretic-validated-20261108'
api.create_repo(DEST,private=True,exist_ok=True)
print('Downloading pinned original files for CPU-only Hub transfer',flush=True)
snapshot=Path(snapshot_download(SRC,revision=REV,max_workers=4))
stage=Path('/kaggle/temp/validated_baseline_upload');stage.mkdir(exist_ok=True)
for src in snapshot.rglob('*'):
    if not src.is_file() or src.name=='README.md':continue
    target=stage/src.relative_to(snapshot);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(src.resolve())
(stage/'UPSTREAM_README.md').write_bytes((snapshot/'README.md').read_bytes())
(stage/'README.md').write_text("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3.5-9B\nlibrary_name: transformers\n---\n# Evaluated Qwen3.5-9B Heretic baseline\n\nAn unchanged copy of [trohrbaugh/Qwen3.5-9B-heretic-v2](https://huggingface.co/trohrbaugh/Qwen3.5-9B-heretic-v2), revision `81dda62421c43a5bd5722edbc6cd47714673dc7f`. Weight-object SHA-256 hashes were checked against that revision. Original weights and precision are preserved; attribution and upstream card are in UPSTREAM_README.md.\n\nOur Kaggle T4 x2 evaluation casts these weights to FP16, disables thinking using the model's template, and uses greedy generation capped at 100 tokens and the original Heretic refusal-marker scorer:\n\n| Condition | Refusals / 100 | Benign first-token KL |\n| --- | ---: | ---: |\n| Original Qwen | 100 | 0 |\n| This upstream checkpoint, measured | 8 | 0.078430 |\n| Upstream card, reported | 6 | 0.0793 |\n| Our rounded-parameter weight replay | 72 | 0.434047 |\n\nThe upstream effect is approximately reproduced; the rounded-parameter replay is not. All 100 upstream-checkpoint answers reached the token limit. Marker absence is not a measure of harmful compliance. The external 0/400 HarmBench result was not tested.\n\n[Full reproduction source, failed replay checkpoint, responses, vectors and provenance](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-replay-20261107). Use this evaluated upstream checkpoint as the fixed baseline for later experiments; keep the failed replay as a diagnostic control.\n")
metrics=json.loads('{"condition": "published_checkpoint_fp16", "model": {"id": "trohrbaugh/Qwen3.5-9B-heretic-v2", "revision": "81dda62421c43a5bd5722edbc6cd47714673dc7f"}, "baseline": {"refusals": 100, "n": 100, "hit_token_limit": 59}, "rounded_replay": {"refusals": 72, "n": 100, "hit_token_limit": 91, "benign_first_token_kl": 0.4340466260910034}, "published_checkpoint_measured": {"refusals": 8, "n": 100, "hit_token_limit": 100, "benign_first_token_kl": 0.07842996716499329}, "published_card": {"baseline_refusals": 100, "edited_refusals": 6, "evaluation_n": 100, "benign_first_token_kl": 0.0793}, "scope": "100-prompt marker scorer, 100-token greedy non-thinking FP16; not HarmBench"}')
(stage/'evaluation').mkdir(exist_ok=True)
(stage/'evaluation/published_checkpoint_diagnostic.json').write_text(json.dumps(metrics,indent=2))
print('Uploading unchanged files; weights are not loaded or converted',flush=True)
api.upload_folder(repo_id=DEST,folder_path=str(stage),commit_message='Preserve evaluated upstream baseline with unchanged original weights')
a=api.model_info(SRC,revision=REV,files_metadata=True)
b=api.model_info(DEST,files_metadata=True)
def hashes(info):
    return {f.rfilename:f.lfs.sha256 for f in info.siblings if f.rfilename.endswith('.safetensors')}
assert hashes(a) and hashes(a)==hashes(b)
proof={'source':SRC,'source_revision':REV,'destination':DEST,'destination_revision':b.sha,
       'private':b.private,'weight_sha256':hashes(b),'all_weight_objects_match':True,
       'copy_method':'CPU-only Kaggle file transfer; no model execution or dtype conversion'}
path=Path('/kaggle/working/baseline_persistence.json');path.write_text(json.dumps(proof,indent=2))
api.upload_file(repo_id=DEST,path_or_fileobj=str(path),path_in_repo='evaluation/baseline_persistence.json',commit_message='Record source-matching weight hashes')
print('Validated baseline persisted:', 'https://huggingface.co/'+DEST,flush=True)
print(json.dumps(proof),flush=True)
