"""Host-only Anthropic Messages forwarding with per-run output-token admission.

No prompts, responses, private thinking, API keys or auth headers enter this
ledger. Numerical usage is observed from native responses; missing usage blocks
new model calls. A completed response may cross the cap; the next is denied.
"""
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlsplit
class ModelProxy(AbstractContextManager):
    def __init__(self,base,out,cap=None):
        url=urlsplit(base)
        if url.scheme not in ('http','https') or url.username or url.password or url.query:
            raise ValueError('Use a plain HTTP(S) base URL without embedded credentials or query')
        self.base=base.rstrip('/');self.out=Path(out);self.cap=cap
        self.key=os.environ.get('ANTHROPIC_API_KEY','local-experiment')
        self.used=0;self.unknown=False;self.lock=threading.Lock();self.admission=threading.BoundedSemaphore(1)
    def __enter__(self):
        self.out.mkdir(parents=True,exist_ok=False);owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                data=json.dumps(dict(ready=not owner.unknown and (owner.cap is None or owner.used<owner.cap),
                    output_tokens=owner.used,output_token_cap=owner.cap)).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                counted=self.path.rstrip('/').endswith('/messages')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=64*1024*1024:self.send_error(413);return
                body=self.rfile.read(size)
                if counted:owner.admission.acquire()
                with owner.lock:
                    blocked=counted and (owner.unknown or (owner.cap is not None and owner.used>=owner.cap))
                if blocked:
                    owner.admission.release();self.send_error(429,'Per-run output budget exhausted or usage unknown');return
                path=self.path
                if owner.base.endswith('/v1') and path.startswith('/v1/'):path=path[3:]
                headers={k:v for k,v in self.headers.items() if k.lower() in
                    ['content-type','anthropic-version','anthropic-beta','accept','x-experiment-episode']}
                headers['x-api-key']=owner.key;headers['Authorization']='Bearer '+owner.key
                request=urllib.request.Request(owner.base+path,data=body,headers=headers,method='POST')
                started=time.monotonic();usage={};complete=False;status=0;failure=None
                try:
                    upstream=urllib.request.urlopen(request,timeout=10800)
                    status=upstream.status;self.send_response(status)
                    typ=upstream.headers.get('Content-Type','application/json')
                    self.send_header('Content-Type',typ);self.send_header('Connection','close');self.end_headers()
                    if 'text/event-stream' in typ:
                        broken=False
                        for line in upstream:
                            if line.startswith(b'data: '):
                                try:
                                    event=json.loads(line[6:])
                                    usage.update(event.get('message',{}).get('usage',{}))
                                    usage.update(event.get('usage',{}))
                                    if event.get('type')=='message_stop':complete=True
                                except (ValueError,TypeError):pass
                            if not broken:
                                try:self.wfile.write(line);self.wfile.flush()
                                except (BrokenPipeError,ConnectionResetError):broken=True
                    else:
                        raw=upstream.read()
                        try:
                            value=json.loads(raw);usage.update(value.get('usage',{}));complete=True
                        except (ValueError,TypeError):pass
                        try:self.wfile.write(raw)
                        except (BrokenPipeError,ConnectionResetError):pass
                    upstream.close()
                except urllib.error.HTTPError as exc:
                    status=exc.code;failure='upstream_http_'+str(status)
                    try:self.send_error(status,'Upstream model request failed; see usage ledger')
                    except OSError:pass
                except Exception as exc:
                    failure=type(exc).__name__
                    try:self.send_error(502,'Upstream model transport failed')
                    except OSError:pass
                finally:
                    if counted:
                        with owner.lock:
                            measured=isinstance(usage.get('output_tokens'),int)
                            if measured:owner.used+=usage['output_tokens']
                            if status==200 and (not measured or not complete):owner.unknown=True
                            record=dict(at=time.time(),http_status=status,complete=complete,usage=usage,
                                elapsed_seconds=time.monotonic()-started,error=failure,output_total=owner.used)
                            with (owner.out/'usage.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        owner.admission.release()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        return self
    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)
