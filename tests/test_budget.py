import json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from eca.budget import ModelProxy
class Budget(unittest.TestCase):
    def test_usage_admission_and_private_content(self):
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                raw=json.dumps(dict(usage={'input_tokens':7,'output_tokens':11},content=[{'text':'do not archive'}])).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Upstream);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                with ModelProxy('http://127.0.0.1:'+str(server.server_port),Path(folder)/'model',10) as proxy:
                    req=urllib.request.Request(proxy.url+'/v1/messages',data=b'{}')
                    self.assertEqual(urllib.request.urlopen(req).status,200)
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
                    self.assertEqual(error.exception.code,429)
                    ledger=(Path(folder)/'model/usage.jsonl').read_text();self.assertNotIn('do not archive',ledger)
                    self.assertEqual(json.loads(ledger)['usage']['output_tokens'],11)
        finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()
