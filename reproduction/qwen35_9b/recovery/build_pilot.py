"""Assemble a self-contained Kaggle pilot from pinned setup and recovery helpers."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
original=json.loads((ROOT/'Reproduce_Qwen3_5_9B_Heretic.ipynb').read_text())
codes=[''.join(c['source']).replace('qwen35_reproduction','qwen35_recovery')
       for c in original['cells'][1:4]]
helper=(HERE/'evaluation.py').read_text()
body=(HERE/'pilot_body.py').read_text()
range_audit=json.loads((HERE/'range_audit.json').read_text())
direction_audits=[json.loads((HERE/'direction_sample_audit.json').read_text())]
if (HERE/'direction_sample_audit_attention.json').exists():
    direction_audits.append(json.loads((HERE/'direction_sample_audit_attention.json').read_text()))
codes.append('EVALUATION_SOURCE = '+repr(helper)+'\nRANGE_AUDIT = '+repr(range_audit)
             +'\nDIRECTION_AUDITS = '+repr(direction_audits)+'\n'+body)
for code in codes:ast.parse(code)
dest=HERE/'kaggle-pilot'
dest.mkdir(exist_ok=True)
script='\n\n'.join(codes)+'\n'
(dest/'recovery_pilot.py').write_text(script)
cells=[{'cell_type':'markdown','metadata':{},'id':'recovery-intro','source':[
    '# Qwen3.5-9B recovery pilot\n',
    'Kaggle T4 x2. Reuse our pinned400+400 activation means; vary only good-mean direction projection. Evaluation stays nonthinking.\n',
    'Two candidates, development-only selection. Historical tests are exposed; fresh subset is not full HarmBench.\n']}]
for code in codes:
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'id':hashlib.sha256(code.encode()).hexdigest()[:12],'source':code.splitlines(keepends=True)})
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
(HERE/'Recover_Qwen3_5_9B_Heretic.ipynb').write_text(json.dumps(notebook,indent=1))
(dest/'Recover_Qwen3_5_9B_Heretic.ipynb').write_text(json.dumps(notebook,indent=1))
(dest/'kernel-metadata.json').write_text(json.dumps({
    'id':'ommprakash007/qwen35-recovery-projected-pilot','title':'Qwen35 Recovery Projected Pilot',
    'code_file':'Recover_Qwen3_5_9B_Heretic.ipynb','language':'python','kernel_type':'notebook','is_private':True,
    'enable_gpu':True,'enable_internet':True,
    'dataset_sources':['ommprakash007/qwen35-phase1-private-runtime-input'],
    'kernel_sources':[],
    'competition_sources':[],'model_sources':[]},indent=2))
print('Prepared pilot; not submitted.',len(script),'bytes')
