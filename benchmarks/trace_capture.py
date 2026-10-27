"""Capture-only foreground costs and HTTP connection reuse; no external requests."""
import json,socket,statistics,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from saina import TraceClient


def stats(values):
    ordered=sorted(values)
    return {name:round(ordered[min(len(ordered)-1,int(len(ordered)*p))],4) for name,p in [('p50_ms',.5),('p95_ms',.95),('p99_ms',.99)]}


def main():
    rows=[]
    for size,n in [(1024,2000),(65536,200),(524288,40)]:
        client=TraceClient(collection='bench',max_pending=5000,flush_interval=3600,on_error=lambda e:None)
        client._worker=lambda:None
        client._post=lambda path,body,**kw:{'decision_ids':[r['decision_id'] for r in body['traces']]}
        payload={'message':'x'*size};times=[]
        for _ in range(n):
            start=time.perf_counter()
            with client.decision(task='route',inputs=payload) as trace:trace.set_answer('billing')
            times.append((time.perf_counter()-start)*1000)
        start=time.perf_counter();client.close();flush_ms=(time.perf_counter()-start)*1000
        assert client.pending_count==0 and client.dropped_count==0
        rows.append({'payload_bytes':size,'samples':n,'capture':stats(times),'flush_cpu_ms':round(flush_ms,3)})
    connections=set()
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*args):pass
        def do_POST(self):
            connections.add(self.client_address[1]);self.rfile.read(int(self.headers['Content-Length']))
            data=b'{"decision_id":"native","collection_id":"bench"}'
            self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=TraceClient(collection='bench',base_url='http://127.0.0.1:'+str(server.server_port));times=[]
    try:
        for _ in range(100):
            start=time.perf_counter();client.invoke_endpoint('test-endpoint',{'state':'ticket'});times.append((time.perf_counter()-start)*1000)
        client.close()
    finally:server.shutdown();server.server_close();thread.join()
    print(json.dumps({'capture':rows,'native_http':{'requests':100,'connections':len(connections),**stats(times)}},indent=2))
if __name__=='__main__':main()
