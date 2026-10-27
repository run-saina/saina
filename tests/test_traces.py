import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from saina import TraceClient, TraceDeliveryError


class TraceCapture(unittest.TestCase):
    def client(self, **kwargs):
        errors=[]
        c=TraceClient(collection='support',flush_interval=3600,on_error=errors.append,**kwargs)
        self.addCleanup(c._stop.set);self.addCleanup(c._wake.set)
        c.errors=errors
        return c

    @staticmethod
    def ack(path, body, **kw):
        return {'decision_ids':[r['decision_id'] for r in body['traces']]}

    def test_explicit_payload_snapshot_and_no_inference_or_network_during_scope(self):
        c=self.client();inputs={'message':'Ticket'}
        with patch.object(c,'_post',side_effect=self.ack) as post:
            with c.decision(task='route-ticket',inputs=inputs) as trace:
                answer={'route':'billing'};trace.set_answer(answer)
            inputs['message']='changed';answer['route']='changed'
            self.assertEqual(post.call_count,0)
            self.assertEqual(c.pending_count,1)
            c.close()
            record=post.call_args.args[1]['traces'][0]
            self.assertEqual(record['inputs'],{'message':'Ticket'})
            self.assertEqual(record['answer'],{'route':'billing'})
            self.assertEqual(record['status'],'completed')
            self.assertEqual(c.pending_count,0)
            with self.assertRaises(RuntimeError):trace.set_answer('late')

    def test_application_exception_is_preserved_without_sensitive_message(self):
        c=self.client()
        error=RuntimeError('sensitive contents')
        try:
            with c.decision(task='route',inputs={}) as trace:
                raise error
        except RuntimeError as caught:self.assertIs(caught,error)
        with patch.object(c,'_post',side_effect=self.ack) as post:
            c.close();record=post.call_args.args[1]['traces'][0]
            self.assertEqual(record['error'],{'type':'RuntimeError'})
            self.assertNotIn('sensitive contents',json.dumps(record))

    def test_failed_delivery_retains_exact_payload_and_bad_receipt_does_not_drop(self):
        c=self.client()
        with c.decision(task='route',inputs={}) as t:t.set_answer(None)
        with patch.object(c,'_post',side_effect=TraceDeliveryError('unavailable')):
            with self.assertRaises(TraceDeliveryError):c.flush()
        self.assertEqual(c.pending_count,1)
        with patch.object(c,'_post',return_value={'decision_ids':[]}):
            with self.assertRaises(TraceDeliveryError):c.flush()
        with patch.object(c,'_post',side_effect=self.ack) as post:
            c.close();self.assertEqual(post.call_args.args[1]['traces'][0]['decision_id'],t.decision_id)

    def test_overflow_is_visible_without_replacing_business_return(self):
        c=self.client(batch_size=100,max_pending=100)
        # Disable delivery to exercise the queue limit without any external network.
        with patch.object(c,'_worker',return_value=None):
            for _ in range(101):
                with c.decision(task='route',inputs={}):pass
        self.assertEqual(c.dropped_count,1);self.assertTrue(c.errors)
        with patch.object(c,'_post',side_effect=self.ack):c.close()

    def test_incomplete_scope_and_json_validation(self):
        c=self.client()
        with self.assertRaises(ValueError):c.decision(task='route',inputs={'invalid':float('nan')})
        with c.decision(task='route',inputs=False):pass
        with patch.object(c,'_post',side_effect=self.ack) as post:
            c.close();record=post.call_args.args[1]['traces'][0]
            self.assertEqual(record['status'],'incomplete');self.assertNotIn('answer',record)

    def test_native_endpoint_is_explicit_and_has_no_external_duplicate(self):
        c=self.client()
        with patch.object(c,'_post',return_value={'decision_id':'native'}) as post:
            result=c.invoke_endpoint('support-triage',{'state':'Ticket'},version=3)
            self.assertEqual(result['decision_id'],'native')
            self.assertEqual(post.call_args.kwargs,{'retry':False})
            self.assertEqual(post.call_args.args[1],{'state':'Ticket','version':3,'trace_collection_id':'support'})
            self.assertEqual(c.pending_count,0)
        c.close()

    def test_async_scopes_and_outcome_flush(self):
        c=self.client()
        calls=[]
        def send(path,body,**kw):
            calls.append((path,body));return self.ack(path,body) if 'traces' in body else {'id':'outcome'}
        async def run():
            async with c:
                async with c.decision(task='route',inputs={'x':1}) as t:t.set_answer('billing')
                await c.areport_outcome(t.decision_id,values={'resolved':True},event_id='resolved1')
        with patch.object(c,'_post',side_effect=send):asyncio.run(run())
        self.assertEqual(len(calls),2);self.assertTrue(calls[1][0].endswith('/outcomes'))

    def test_real_http_retry_keeps_event_ids_and_authorization(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path,body,self.headers.get('Authorization')))
                status=503 if len(requests)==1 else 201
                value={'error':{'code':'overloaded'}} if status==503 else {'decision_ids':[r['decision_id'] for r in body['traces']]}
                data=json.dumps(value).encode();self.send_response(status);self.send_header('Retry-After','0');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            c=self.client(base_url='http://127.0.0.1:'+str(server.server_port),api_key='test-token')
            with c.decision(task='route',inputs={'message':'Ticket'}) as t:t.set_answer('billing')
            c.close()
            self.assertEqual(len(requests),2);self.assertEqual(requests[0],requests[1])
            self.assertEqual(requests[0][2],'Bearer test-token')
        finally:server.shutdown();server.server_close();thread.join()

    def test_native_connection_pool_is_bounded_and_reuses_sockets_under_concurrency(self):
        from concurrent.futures import ThreadPoolExecutor
        import time,socket
        peers=set();active=maximum=count=0;lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def setup(self):super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            def log_message(self,*args):pass
            def do_POST(self):
                nonlocal active,maximum,count
                self.rfile.read(int(self.headers['Content-Length']))
                with lock:peers.add(self.client_address[1]);active+=1;maximum=max(maximum,active);count+=1
                time.sleep(.01)
                data=b'{"decision_id":"native","collection_id":"support"}'
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                with lock:active-=1
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=self.client(base_url='http://127.0.0.1:'+str(server.server_port),max_connections=4)
        try:
            with ThreadPoolExecutor(max_workers=16) as executor:
                results=list(executor.map(lambda i:client.invoke_endpoint('support-triage',{'state':str(i)}),range(64)))
            self.assertEqual(len(results),64);self.assertEqual(count,64);self.assertLessEqual(maximum,4);self.assertLessEqual(len(peers),4)
            client.close()
        finally:server.shutdown();server.server_close();thread.join()

    def test_native_disconnect_is_not_replayed_and_terminal_html_errors_do_not_retry(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                requests.append(self.path);self.rfile.read(int(self.headers['Content-Length']))
                if self.path.endswith('/decide'):
                    self.connection.shutdown(2);self.connection.close();return
                data=b'<html>not authorized</html>'
                self.send_response(403);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=self.client(base_url='http://127.0.0.1:'+str(server.server_port))
        try:
            with self.assertRaises(TraceDeliveryError):client.invoke_endpoint('support-triage',{'state':'Ticket'})
            self.assertEqual(len(requests),1)
            with self.assertRaises(TraceDeliveryError) as error:client._post('/rejected',{})
            self.assertEqual(error.exception.status,403);self.assertEqual(len(requests),2)
            client.close()
        finally:server.shutdown();server.server_close();thread.join()

    def test_native_outcome_retries_temporary_missing_evidence_without_reinvoking(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path,body))
                status=404 if len(requests)==2 else 200
                data=json.dumps({'decision_id':'native'} if self.path.endswith('/decide') else {'id':'outcome'}).encode()
                self.send_response(status);self.send_header('Content-Length',str(len(data)));self.send_header('Retry-After','0');self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client=self.client(base_url='http://127.0.0.1:'+str(server.server_port))
        try:
            decision=client.invoke_endpoint('support-triage',{'state':'Ticket'})
            client.report_outcome(decision['decision_id'],values={'resolved':True},event_id='resolved')
            self.assertEqual(sum(p.endswith('/decide') for p,b in requests),1)
            self.assertEqual(requests[1],requests[2]);client.close()
        finally:server.shutdown();server.server_close();thread.join()
