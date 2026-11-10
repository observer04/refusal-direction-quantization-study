"""Stdlib-only private small-artifact persistence. Never downloads model weights."""
import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO='Observer04/Qwen3.5-9B-heretic-corrected-20261109'
PREFIX='research/model_audit_v1/review-v1/'
def token():
    if os.environ.get('HF_TOKEN'):return os.environ['HF_TOKEN']
    for line in (ROOT/'.env').read_text().splitlines():
        k,sep,v=line.strip().partition('=')
        if sep and k=='HF_TOKEN':return v.strip().strip('"\'')
    raise RuntimeError('Workspace HF credential missing')
def request(url,credential,data=None):
    headers={'Authorization':'Bearer '+credential}
    if data is not None:headers['Content-Type']='application/x-ndjson'
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=120) as r:return json.load(r)
def roots(info):return {r['rfilename']:r for r in info['siblings'] if '/' not in r['rfilename']}
def main():
    credential=token()
    info=request('https://huggingface.co/api/models/'+REPO+'?blobs=true',credential)
    assert info['private'],'Destination must be private'
    baseline=json.loads((HERE/'remote_provenance_audit.json').read_text())['models']['corrected']['files']
    compact=lambda rs:{r['rfilename']:{k:r[k] for k in ['size','blobId','lfs'] if k in r} for r in rs if '/' not in r['rfilename']}
    assert compact(info['siblings'])==compact(baseline),'Root files differ from pinned corrected checkpoint'
    assert not any(r['rfilename'].startswith(PREFIX) for r in info['siblings']),'Versioned path already exists; do not overwrite'
    files={}
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ['persistence_verification.json','artifact_manifest.json']:continue
        if p.suffix not in ['.py','.md','.json','.jsonl','.html','.csv','.log','.txt']:continue
        blob=p.read_bytes()
        assert len(blob)<10_000_000,(p,'Unexpected artifact size')
        assert credential.encode() not in blob,(p,'Credential detected')
        assert p.suffix not in ['.safetensors','.bin','.pt','.pth'], 'No weights or model runtime artifacts'
        files[str(p.relative_to(HERE))]=blob
    digest={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}
    files['artifact_manifest.json']=json.dumps(digest,indent=2).encode()
    (HERE/'artifact_manifest.json').write_bytes(files['artifact_manifest.json'])
    payload=[{'key':'header','value':{'summary':'Persist independent model audit, exact metrics, every completion and separate human review',
                                   'parentCommit':info['sha']}}]
    payload.extend({'key':'file','value':{'path':PREFIX+n,'encoding':'base64','content':base64.b64encode(b).decode()}} for n,b in files.items())
    result=request('https://huggingface.co/api/models/'+REPO+'/commit/main',credential,
                   ''.join(json.dumps(x)+'\n' for x in payload).encode())
    assert result['success']
    after=request('https://huggingface.co/api/models/'+REPO+'/revision/'+result['commitOid']+'?blobs=true',credential)
    assert after['private'] and roots(info)==roots(after),'Root checkpoint files changed'
    remote={r['rfilename']:r for r in after['siblings']}
    assert all(PREFIX+n in remote for n in files)
    git_verified=0
    for name,blob in files.items():
        entry=remote[PREFIX+name]
        if entry.get('blobId') and not entry.get('lfs'):
            expected=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
            assert entry['blobId']==expected,(name,'Remote git blob hash mismatch')
            git_verified+=1
    # Verify representative artifact bytes directly at the immutable commit, with no weight downloads.
    verified={}
    for name in ['AUDIT.md','PROTOCOL.md','artifact_manifest.json','new_blind_labels.jsonl','classifier_comparison_all.json']:
        if name not in files:continue
        u='https://huggingface.co/'+REPO+'/resolve/'+result['commitOid']+'/'+PREFIX+name
        with urllib.request.urlopen(urllib.request.Request(u,headers={'Authorization':'Bearer '+credential}),timeout=60) as r:b=r.read()
        assert b==files[name];verified[name]=hashlib.sha256(b).hexdigest()
    report={'repo':REPO,'private':True,'commit':result['commitOid'],'path':PREFIX,'artifact_count':len(files),
            'artifact_bytes':sum(map(len,files.values())),'root_files_unchanged':True,'root_files_equal_pinned_corrected':True,'representative_remote_bytes_verified':verified}
    report['remote_git_blob_hashes_verified']=git_verified
    (HERE/'persistence_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
