"""Tiny direction fingerprint using remote weight bytes and saved float arrays.

Stdlib only; no pickle execution, tensor library, model inference or full shards.
"""
import json
import math
from pathlib import Path
import pickletools
import struct
import sys
import zipfile
import audit_ranges as ar
import concurrent.futures

HERE=Path(__file__).resolve().parent
NAME=(sys.argv[1] if len(sys.argv)>1 else 'model.language_model.layers.20.mlp.down_proj.weight')
ROWS=list(range(0,4096,128))
COLS=128

def unit(values):
    norm=math.sqrt(sum(x*x for x in values))
    if norm==0:
        return [0. for _ in values]
    return [x/norm for x in values]

def dot(a,b):return sum(x*y for x,y in zip(a,b))

def saved_arrays():
    path=HERE.parent/'runs/batch-v1/qwen35_reproduction/results/refusal_directions.pt'
    with zipfile.ZipFile(path) as z:
        prefix=next(n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/data.pkl'))
        ops=list(pickletools.genops(z.read(prefix+'/data.pkl')))
        strings=[arg for op,arg,_ in ops if op.name=='BINUNICODE']
        assert strings==['good_means','storage','0','cpu','bad_means','1','directions','2'],strings
        assert ('GLOBAL','torch FloatStorage') in [(op.name,arg) for op,arg,_ in ops]
        arrays=[]
        for key in ('0','2'):
            blob=z.read(prefix+'/data/'+key)
            assert len(blob)==33*4096*4
            values=[v[0] for v in struct.iter_unpack('<f',blob)]
            arrays.append([values[i*4096:(i+1)*4096] for i in range(33)])
    return arrays

def run():
    headers={}
    for label,key in [('base','base_model'),('published','reference_model')]:
        source=ar.SOURCES[key]
        index=ar.small_json(f"https://huggingface.co/{source['id']}/resolve/{source['revision']}/model.safetensors.index.json")['weight_map']
        _,headers[label]=ar.read_header((label,source,index[NAME]))
    def row_values(item):
        label,row=item
        h=headers[label]
        entry=h['header'][NAME]
        assert entry['dtype']=='BF16' and entry['shape'][0]==4096,entry
        start=h['offset']+entry['data_offsets'][0]+row*entry['shape'][1]*2
        return (label,row),ar.floats(ar.read_range(h['url'],start,start+COLS*2-1),'BF16')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        samples=dict(pool.map(row_values,[(label,row) for row in ROWS for label in ('base','published')]))
    D=[[b-a for a,b in zip(samples['base',row],samples['published',row])] for row in ROWS]
    v=unit([sum(row) for row in D])
    for _ in range(5):
        a=[sum(v[r]*D[r][c] for r in range(len(ROWS))) for c in range(COLS)]
        v=unit([dot(row,a) for row in D])
    a=[sum(v[r]*D[r][c] for r in range(len(ROWS))) for c in range(COLS)]
    rank1_energy=dot(a,a)/sum(dot(row,row) for row in D)
    means,directions=saved_arrays()
    projected=[]
    for mean,direction in zip(means,directions):
        good=unit(mean)
        coeff=dot(direction,good)
        projected.append(unit([d-coeff*g for d,g in zip(direction,good)]))
    def interpolate(vectors):return unit([.51*a+.49*b for a,b in zip(vectors[19],vectors[20])])
    comparisons={}
    for label,vectors in [('raw',directions),('projected',projected)]:
        selected=interpolate(vectors)
        selected_cos=abs(dot(v,unit([selected[r] for r in ROWS])))
        slot_cosines=[abs(dot(v,unit([row[r] for r in ROWS]))) for row in vectors]
        comparisons[label]={'selected_abs_cosine_32_coordinates':selected_cos,
                            'best_hidden_slot':max(range(33),key=lambda i:slot_cosines[i]),
                            'best_slot_abs_cosine':max(slot_cosines),'slot_cosines':slot_cosines}
    output={'name':NAME,'sampled_output_rows':ROWS,'sampled_input_columns':list(range(COLS)),
            'remote_bytes_including_headers':ar.READ_BYTES,'sampled_delta_dominant_direction':v,
            'sampled_delta_rank1_energy_fraction':rank1_energy,'comparisons':comparisons,
            'limits':['Only32outputcoordinates and128inputcolumns sampled.',
                      'Direction alignment is a diagnostic fingerprint, not a full-vector equality test.',
                      'No tensors/model/dependency environment executed locally.']}
    filename='direction_sample_audit_attention.json' if len(sys.argv)>1 else 'direction_sample_audit.json'
    (HERE/filename).write_text(json.dumps(output,indent=2))
    print(json.dumps({k:output[k] for k in ('remote_bytes_including_headers','sampled_delta_rank1_energy_fraction','comparisons')}))

if __name__=='__main__':run()
