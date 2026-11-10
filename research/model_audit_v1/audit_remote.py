"""Metadata and small text only. Never download full model weights locally."""
import hashlib,json,os,urllib.request
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
def credential():
 if os.environ.get('HF_TOKEN'):return os.environ['HF_TOKEN']
 for s in (R/'.env').read_text().splitlines():
  k,sep,v=s.strip().partition('=')
  if sep and k=='HF_TOKEN':return v.strip().strip('"\'')
 raise RuntimeError('HF_TOKEN absent')
token=credential()
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=60) as r:return r.read()
pins=json.loads((H/'pins.json').read_text());results={}
for name in ['original','publisher','corrected']:
 pin=pins[name];info=json.loads(get('https://huggingface.co/api/models/'+pin['id']+'/revision/'+pin['revision']+'?blobs=true'));assert info['sha']==pin['revision']
 results[name]={'id':pin['id'],'revision':info['sha'],'private':info.get('private',False),'files':[{k:s[k] for k in ['rfilename','size','blobId','lfs'] if k in s} for s in info['siblings'] if '/' not in s['rfilename']]}
 for filename in ['README.md','config.json','generation_config.json','tokenizer_config.json','chat_template.jinja','model.safetensors.index.json']:
  try:b=get('https://huggingface.co/'+pin['id']+'/resolve/'+pin['revision']+'/'+filename)
  except Exception as exc:results[name].setdefault('text_access_errors',{})[filename]=type(exc).__name__;continue
  assert len(b)<1000000
  p=H/'reference'/name/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  results[name].setdefault('small_text_sha256',{})[filename]=hashlib.sha256(b).hexdigest()
corrected=pins['corrected'];latest=json.loads(get('https://huggingface.co/api/models/'+corrected['id']+'?blobs=true'));assert latest['private']
roots=lambda info:{s['rfilename']:{k:s[k] for k in ['size','blobId','lfs'] if k in s} for s in info['siblings'] if '/' not in s['rfilename']}
assert roots(latest)=={s['rfilename']:{k:s[k] for k in ['size','blobId','lfs'] if k in s} for s in results['corrected']['files']}
report={'models':results,'latest_corrected_revision_at_check':latest['sha'],'corrected_root_files_unchanged_from_pinned':True,'access_scope':'Only API file metadata and <=1MB model text/index files. No local weights.'};(H/'remote_provenance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'latest_corrected_revision':latest['sha'],'root_unchanged':True,'model_pins_verified':3,'original_publisher_chat_template_equal':results['original']['small_text_sha256']['chat_template.jinja']==results['publisher']['small_text_sha256']['chat_template.jinja']},indent=2))
