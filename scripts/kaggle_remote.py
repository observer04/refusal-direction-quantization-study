"""Run notebook cells on the user's existing Kaggle Jupyter kernel.

Uses the VS Code Jupyter connection cache; never prints connection credentials.
No kernel is created. Local execution is only this lightweight transport client.
"""
import argparse
import base64
import datetime as dt
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

import websocket

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/home/op/.config/Code/User/globalStorage/ms-toolsai.jupyter/remoteKernelSpecCache.json')


def connection():
    data = json.loads(CACHE.read_text())
    entries = [e for es in data['data'].values() for e in es
               if e.get('kind') == 'connectToLiveRemoteKernel']
    if not entries:
        raise RuntimeError('No existing VS Code remote kernel connection found')
    entry = entries[-1]
    return entry['baseUrl'].rstrip('/'), entry['kernelModel']['id']


def request(base, endpoint, method='GET'):
    for attempt in range(3):
        try:
            req = urllib.request.Request(base + '/' + endpoint, method=method)
            with urllib.request.urlopen(req, timeout=25) as response:
                return json.load(response) if response.status != 204 else None
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError('Kaggle REST request failed: ' + type(exc).__name__) from None
            time.sleep(2)


class Kernel:
    def __init__(self):
        self.base, self.kernel_id = connection()
        self.session = uuid.uuid4().hex
        url = self.base.replace('https://', 'wss://').replace('http://', 'ws://')
        url += '/api/kernels/' + self.kernel_id + '/channels?session_id=' + self.session
        for attempt in range(4):
            try:
                self.ws = websocket.create_connection(url, timeout=30, suppress_origin=True)
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError('Kaggle WebSocket failed: ' + type(exc).__name__) from None
                time.sleep(3)

    def execute(self, code, on_output=None, private=False):
        msg_id = uuid.uuid4().hex
        msg = {'header': {'msg_id': msg_id, 'username': 'phase1-client',
                         'session': self.session, 'date': dt.datetime.now(dt.timezone.utc).isoformat(),
                         'msg_type': 'execute_request', 'version': '5.3'},
               'parent_header': {}, 'metadata': {}, 'channel': 'shell',
               'content': {'code': code, 'silent': private, 'store_history': not private,
                           'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True},
               'buffers': []}
        self.ws.send(json.dumps(msg))
        reply, idle = None, False
        while not (reply is not None and idle):
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                self.ws.ping()
                continue
            if not raw:
                raise RuntimeError('Kaggle kernel channel closed')
            if isinstance(raw, bytes):
                raise RuntimeError('Unexpected binary kernel message')
            event = json.loads(raw)
            if event.get('parent_header', {}).get('msg_id') != msg_id:
                continue
            kind, content = event['header']['msg_type'], event.get('content', {})
            if kind == 'execute_reply':
                reply = content
            elif kind == 'status' and content.get('execution_state') == 'idle':
                idle = True
            elif on_output and kind in ('stream', 'display_data', 'execute_result', 'error', 'update_display_data', 'execute_input'):
                on_output(kind, content)
        return reply

    def close(self):
        self.ws.close()


def inject_hf(kernel):
    values = {}
    for line in (ROOT / '.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip().removeprefix('export ')] = value.strip().strip('\"\'')
    token = values.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN missing from .env')
    # Silent, unrecorded execution: token is not placed in notebook source/history/output.
    reply = kernel.execute('import os\nos.environ["HF_TOKEN"] = ' + repr(token), private=True)
    if reply.get('status') != 'ok':
        raise RuntimeError('Could not pass HF credentials to the Kaggle process')


def sync_source(kernel):
    files = [ROOT / 'Reproduce_Qwen3_5_9B_Heretic.ipynb', ROOT / 'STATE.md',
             ROOT / 'scripts/kaggle_remote.py', ROOT / 'scripts/build_qwen35_notebook.py']
    material = ROOT / 'reproduction/qwen35_9b'
    files += [material / name for name in ['README.md','sources.json','published_parameters.json',
               'requirements-kaggle.txt','reference/published_model_card.md',
               'reference/heretic-qwen35-source.zip']]
    payload = {str(p.relative_to(ROOT)):base64.b64encode(p.read_bytes()).decode() for p in files if p.exists()}
    code = "import base64, json\nfrom pathlib import Path\n_payload = json.loads(" + repr(json.dumps(payload)) + ")\n"
    code += "for _name,_data in _payload.items():\n _dest=Path('/kaggle/working/qwen35_reproduction/results/reproduction_source')/_name\n _dest.parent.mkdir(parents=True,exist_ok=True)\n _dest.write_bytes(base64.b64decode(_data))\ndel _payload,_data\n"
    reply = kernel.execute(code, private=True)
    if reply.get('status') != 'ok': raise RuntimeError('Source sync failed')
    print('Whitelisted reproduction source synced; credentials excluded.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--code-file', type=Path)
    parser.add_argument('--notebook', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--inject-hf', action='store_true')
    parser.add_argument('--sync-source', action='store_true')
    args = parser.parse_args()
    kernel = Kernel()
    try:
        if args.inject_hf:
            inject_hf(kernel)
            print('HF credential passed privately to the existing Kaggle process.', flush=True)
        if args.sync_source:
            sync_source(kernel)
        if args.probe:
            code = '''import json, os, shutil, subprocess
from pathlib import Path
print(json.dumps({'kaggle':Path('/kaggle').exists(),'cwd':os.getcwd(),'free_disk_gib':round(shutil.disk_usage('/kaggle/working').free/2**30,1)}))
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)
print('existing_model', 'model' in globals())
'''
            reply = kernel.execute(code, lambda kind, c: print(c.get('text', c.get('ename', '')), end='', flush=True))
            print('\nProbe:', reply.get('status'), flush=True)
        if args.code_file:
            def output(kind, content):
                if kind == 'stream': print(content['text'], end='', flush=True)
                elif kind == 'error': print('\n'.join(content.get('traceback', [])), flush=True)
                elif kind in ('execute_result', 'display_data'):
                    print(content.get('data', {}).get('text/plain', ''), flush=True)
            reply = kernel.execute(args.code_file.read_text(), output)
            print('Execution:', reply.get('status'), flush=True)
            if reply.get('status') != 'ok': sys.exit(1)
        if args.notebook:
            nb = json.loads(args.notebook.read_text())
            executed_path = args.notebook.with_name(args.notebook.stem + '.executed.ipynb')
            if executed_path.exists():
                old = json.loads(executed_path.read_text())
                for i, cell in enumerate(nb['cells']):
                    if i < len(old['cells']) and cell['source'] == old['cells'][i]['source'] and cell['cell_type'] == 'code':
                        cell['outputs'] = old['cells'][i].get('outputs', [])
                        cell['execution_count'] = old['cells'][i].get('execution_count')
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] != 'code' or i < args.start or (args.end is not None and i > args.end):
                    continue
                print(f'[{dt.datetime.now().isoformat(timespec="seconds")}] Cell {i} started', flush=True)
                cell['outputs'] = []
                def output(kind, c):
                    if kind == 'execute_input':
                        cell['execution_count'] = c.get('execution_count'); return
                    if kind == 'update_display_data':
                        return
                    out = dict(c)
                    out.pop('transient', None)
                    out['output_type'] = kind
                    cell['outputs'].append(out)
                    if kind == 'stream':
                        # tqdm writes carriage-return progress; preserve in notebook but keep terminal terse.
                        text = c['text']
                        if '\r' not in text: print(text[-6000:], end='', flush=True)
                    elif kind == 'error':
                        print('\n'.join(c.get('traceback', [])), flush=True)
                reply = kernel.execute(''.join(cell['source']), output)
                executed_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
                print(f'[{dt.datetime.now().isoformat(timespec="seconds")}] Cell {i}: {reply.get("status")}', flush=True)
                if reply.get('status') != 'ok': sys.exit(1)
    finally:
        kernel.close()

if __name__ == '__main__':
    main()
