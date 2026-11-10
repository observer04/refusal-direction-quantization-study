"""Read-only remote byte samples; stdlib only, no model/tensor execution.

Reads safetensors metadata and tiny samples via strictly validated HTTP ranges.
Never downloads complete model shards. Raw weight samples are not retained.
"""
import concurrent.futures
import json
import math
from pathlib import Path
import struct
import threading
import time
import urllib.request

HERE=Path(__file__).resolve().parent
SOURCES=json.loads((HERE.parent/'sources.json').read_text())
LOCK=threading.Lock()
READ_BYTES=0
BYTE_LIMIT=2_000_000

def small_json(url):
    with urllib.request.urlopen(url,timeout=40) as response:
        blob=response.read(250001)
    assert len(blob)<=250000
    return json.loads(blob)

def read_range(url,start,end):
    global READ_BYTES
    expected=end-start+1
    assert 0<expected<250000
    req=urllib.request.Request(url+f'?audit_range={start}-{end}',
                               headers={'Range':f'bytes={start}-{end}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=40) as response:
                assert response.status==206, 'Server did not honor byte range; aborted before shard download.'
                actual=response.headers.get('Content-Range','')
                assert actual.startswith(f'bytes {start}-{end}/'), (start,end,actual)
                blob=response.read(expected+1)
            assert len(blob)==expected
            with LOCK:
                READ_BYTES+=len(blob)
                assert READ_BYTES<BYTE_LIMIT,'Strict aggregate byte budget exceeded.'
            return blob
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

def read_header(item):
    label,source,shard=item
    url=f"https://huggingface.co/{source['id']}/resolve/{source['revision']}/{shard}"
    length=struct.unpack('<Q',read_range(url,0,7))[0]
    assert length<250000
    header=json.loads(read_range(url,8,7+length))
    return (label,shard),{'url':url,'offset':length+8,'header':header}

def floats(blob,dtype):
    if dtype=='BF16':
        return [struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',blob)]
    return [x[0] for x in struct.iter_unpack({'F16':'<e','F32':'<f'}[dtype],blob)]

def cast_bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',rounded))[0]

def run():
    indexes={}
    work=[]
    for label,key in [('base','base_model'),('published','reference_model')]:
        source=SOURCES[key]
        url=f"https://huggingface.co/{source['id']}/resolve/{source['revision']}/model.safetensors.index.json"
        indexes[label]=small_json(url)['weight_map']
        work.extend((label,source,shard) for shard in set(indexes[label].values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        headers=dict(pool.map(read_header,work))
    common=sorted(set(indexes['base'])&set(indexes['published']))
    selected=[name for name in common if '.language_model.layers.' in name and any(
        name.endswith(suffix) for suffix in ['.mlp.down_proj.weight','.self_attn.o_proj.weight',
        '.linear_attn.out_proj.weight','.mlp.gate_proj.weight','.input_layernorm.weight'])]
    selected += ['model.language_model.embed_tokens.weight','lm_head.weight']
    def compare(name):
        entries={}
        for label in ('base','published'):
            h=headers[label,indexes[label][name]]
            entry=h['header'][name]
            size={'BF16':2,'F16':2,'F32':4}[entry['dtype']]
            start=h['offset']+entry['data_offsets'][0]
            n=min(128,math.prod(entry['shape']))
            blob=read_range(h['url'],start,start+n*size-1)
            entries[label]=(entry['dtype'],floats(blob,entry['dtype']))
        assert entries['published'][0]=='BF16'
        a=[cast_bf16(v) for v in entries['base'][1]]
        b=entries['published'][1]
        delta=[y-x for x,y in zip(a,b)]
        return {'name':name,'base_dtype':entries['base'][0],'published_dtype':entries['published'][0],
                'sample_n':len(delta),'changed_after_dtype_match':sum(d!=0 for d in delta),
                'max_abs_delta':max(abs(d) for d in delta)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(compare,selected))
    output={'mode':'Metadata plus128-value remote byte samples; no model/tensor execution',
            'sources':{k:SOURCES[k] for k in ('base_model','reference_model')},
            'total_remote_weight_header_and_sample_bytes':READ_BYTES,
            'strict_aggregate_byte_limit':BYTE_LIMIT,'sampled_tensors':len(rows),
            'changed_samples':[r for r in rows if r['changed_after_dtype_match']],
            'unchanged_samples':[r for r in rows if not r['changed_after_dtype_match']],
            'limit':'Sparse samples do not prove whole-tensor equality or rank/direction.'}
    (HERE/'range_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:output[k] for k in ['mode','total_remote_weight_header_and_sample_bytes','sampled_tensors']}))
    print('Changed tensors:',len(output['changed_samples']))
    print('Projection changes:',[r['name'].split('.layers.')[1] for r in output['changed_samples'] if '.layers.' in r['name']])

if __name__=='__main__':run()
