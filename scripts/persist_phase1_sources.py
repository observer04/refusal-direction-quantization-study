"""Persist only whitelisted Phase 1 source/evidence files; never upload .env."""
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Observer04/Qwen3.5-9B-heretic-replay-20261107'
values = {}
for line in (ROOT / '.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key, value = line.split('=', 1)
        values[key.strip().removeprefix('export ')] = value.strip().strip('\"\'')
headers = {'Authorization': 'Bearer ' + values['HF_TOKEN']}


def call(path, payload=None, content_type='application/json'):
    req = urllib.request.Request('https://huggingface.co' + path, data=payload,
                                 headers={**headers, 'Content-Type': content_type})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode(errors='replace')[:1200].replace(values['HF_TOKEN'],'[redacted]')
        print('Hub response:',exc.code,message)
        raise


try:
    call('/api/repos/create', json.dumps({'name': REPO.split('/')[1], 'type': 'model', 'private': True}).encode())
except urllib.error.HTTPError as exc:
    if exc.code != 409:
        raise
files = [ROOT / 'Reproduce_Qwen3_5_9B_Heretic.ipynb', ROOT / 'Evaluate_Published_Qwen3_5_9B.ipynb', ROOT / 'STATE.md']
files.extend(sorted((ROOT / 'scripts').glob('*.py')))
material = ROOT / 'reproduction/qwen35_9b'
files.extend(material / name for name in ['README.md', 'sources.json', 'published_parameters.json',
             'requirements-kaggle.txt', 'reference/published_model_card.md',
             'kaggle/kernel-metadata.json', 'kaggle-reference/kernel-metadata.json',
             'kaggle-persist/kernel-metadata.json', 'PHASE1_RESULTS.md',
             'baseline_persistence.json', 'phase1_summary.json'])
records = [{'key': 'header', 'value': {'summary': 'Persist Phase 1 reproduction source and current state'}}]
for path in files:
    records.append({'key': 'file', 'value': {
        'path': 'experiments/reproduction_source/' + str(path.relative_to(ROOT)),
        'content': base64.b64encode(path.read_bytes()).decode(), 'encoding': 'base64'}})
result = call('/api/models/' + REPO + '/commit/main',
              '\n'.join(json.dumps(x) for x in records).encode(), 'application/x-ndjson')
print('Sources persisted:', result.get('commitUrl', REPO))
