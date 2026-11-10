"""Recompute saved-vector/factor evidence using stdlib; no model/tensor runtime."""
import hashlib,json,math,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];helper=R/'reproduction/qwen35_9b/recovery';sys.path.insert(0,str(helper))
from verify_recovery_results import read_own_saved_float_arrays,unit,verify_saved_adapter_B
out=helper/'runs/corrected-v1/qwen35_corrected/results';pilot=helper/'runs/pilot-v1/qwen35_recovery/results'
good,bad,vectors=read_own_saved_float_arrays(out/'refusal_directions.pt');errors=[]
for g,b,v in zip(good,bad,vectors):
 raw=unit([y-x for x,y in zip(g,b)]);gd=unit(g);c=math.fsum(x*y for x,y in zip(raw,gd));p=unit([x-c*y for x,y in zip(raw,gd)]);errors.append(max(abs(x-y) for x,y in zip(p,v)))
manifest=json.loads((out/'run_manifest.json').read_text());adapter=out/'adapter';blob=(adapter/'adapter_model.safetensors').read_bytes();assert hashlib.sha256(blob).hexdigest()=='c38fbd7c11c86c9ea898477344b588f5342c7ed95a2ce4c8942335af5f73f0db'
hsize=struct.unpack('<Q',blob[:8])[0];header=json.loads(blob[8:8+hsize]);tensors={n:e for n,e in header.items() if n!='__metadata__'};assert len(tensors)==128 and all(e['dtype']=='F32' for e in tensors.values())
breport=verify_saved_adapter_B(out,vectors,manifest['published_parameters'],adapter_subdir='adapter')
oldg,oldb,oldv=read_own_saved_float_arrays(pilot/'projected_refusal_directions.pt');assert good==oldg and bad==oldb and vectors==oldv
f=R/'research/feasibility_v1/runs/run-v1/output/feasibility_v1/results';merge=json.loads((f/'frozen_delta_merge_verification.json').read_text());assert len(merge['checks'])==64 and all(merge['checks'].values())
kl=json.loads((out/'benign_first_token_kl_terms.json').read_text());assert len(kl['per_prompt_kl'])==100
report={'scope':'Independent stdlib recalculation of saved vectors and B factors; merged-writer inference hashes are audited saved Kaggle witnesses, not locally reloaded weights. No full-weight/model execution.', 'projected_slots':len(vectors),'max_projection_coordinate_error':max(errors),'adapter_sha256':hashlib.sha256(blob).hexdigest(),'factor_tensors':len(tensors),'all_factor_storage_F32':True,'B_factor_check':breport,'fresh_saved_means_equal_pilot':True,'merge_exact_writer_hash_checks':64,'merge_all_writer_hashes_match':True,'benign_first_token_KL_mean_of100savedterms':math.fsum(kl['per_prompt_kl'])/100,'merge_definition':merge['definition']}
assert max(errors)<2e-6
(H/'edit_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
