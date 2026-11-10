"""Label the failed replay honestly and publish final Phase 1 evidence."""
import base64,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO='Observer04/Qwen3.5-9B-heretic-replay-20261107'
values={}
for line in (ROOT/'.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);values[k.strip()]=v.strip().strip('\"\'')
headers={'Authorization':'Bearer '+values['HF_TOKEN']}
def get(path):
    with urllib.request.urlopen(urllib.request.Request('https://huggingface.co/'+REPO+'/resolve/main/'+path,headers=headers),timeout=30) as r:return r.read()
def file(path,data):
    return {'key':'file','value':{'path':path,'encoding':'base64','content':base64.b64encode(data).decode()}}
card='''---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
---
# Qwen3.5-9B rounded-parameter replay — failed reproduction

**These root weights score 72/100 refusals, not 6/100.** They are retained as a diagnostic control.
Use the [evaluated publisher baseline](https://huggingface.co/Observer04/Qwen3.5-9B-heretic-validated-20261108) for the next phase.

| Condition | Refusals / 100 | Benign first-token KL |
| --- | ---: | ---: |
| Original Qwen, measured | 100 | 0 |
| These replay weights, measured | 72 | 0.434047 |
| Actual publisher checkpoint, measured in FP16 | 8 | 0.078430 |
| Publisher's card, reported | 6 | 0.0793 |

The replay was exported, reloaded and checked: all 100 response strings matched the pre-save results exactly.
No optimizer search was run. The author's exact extraction tensors, execution environment and unrounded parameters were unavailable; the construction mismatch remains unresolved.

Evaluation: Kaggle T4 x2, greedy non-thinking FP16 generation, up to 100 new tokens with EOS enabled, original Heretic refusal markers and benign first-token KL. Marker absence is not harmful compliance. All 100 publisher-checkpoint answers hit the token limit. The separate 0/400 HarmBench claim was not tested.

Evidence is in `experiments/`; independent publisher evaluation is in `diagnostics/published_fp16/`.
Final high-level state and results are in `STATE.md` and `PHASE1_RESULTS.md`; source notebooks are retained under `experiments/reproduction_source/`.
The adapter's base-model path/revision metadata was corrected for portability after export; its weights were unchanged.
'''
records=[{'key':'header','value':{'summary':'Finalize Phase 1: label failed replay and preserve validated baseline evidence'}}]
records.append(file('README.md',card.encode()))
for name,path in [('STATE.md',ROOT/'STATE.md'),('PHASE1_RESULTS.md',ROOT/'reproduction/qwen35_9b/PHASE1_RESULTS.md'),('baseline_persistence.json',ROOT/'reproduction/qwen35_9b/baseline_persistence.json'),('phase1_summary.json',ROOT/'reproduction/qwen35_9b/phase1_summary.json')]:
    records.append(file(name,path.read_bytes()))
config=json.loads(get('experiments/adapter/adapter_config.json'))
config['base_model_name_or_path']='Qwen/Qwen3.5-9B'
config['revision']='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
records.append(file('experiments/adapter/adapter_config.json',(json.dumps(config,indent=2)+'\n').encode()))
req=urllib.request.Request('https://huggingface.co/api/models/'+REPO+'/commit/main',data='\n'.join(json.dumps(x) for x in records).encode(),headers={**headers,'Content-Type':'application/x-ndjson'})
with urllib.request.urlopen(req,timeout=60) as r:result=json.load(r)
print('Final evidence persisted:',result['commitUrl'])
