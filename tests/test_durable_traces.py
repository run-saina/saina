import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from saina import Traces, TraceFlushResult, SpoolFullError, TraceStorageError


def recorder(tmp_path, **kwargs):
    return Traces('http://localhost:1', api_key='super-secret-key', spool_path=tmp_path/'spool.db', **kwargs)


def ack(rows):
    return [r['trace_id'] for r in rows], []


def quiet(t):
    # Keep delivery deterministic; log() itself must still really commit to disk.
    t._start = lambda: None
    return t


def stored(path):
    with sqlite3.connect(path) as db:
        return [(json.loads(p),state,error) for p,state,error in db.execute('SELECT payload,state,error FROM records ORDER BY seq')]


def test_lazy_commit_snapshot_omission_and_closed_behavior(tmp_path):
    t = quiet(recorder(tmp_path))
    assert not t.spool_path.exists()
    assert t.pending == 0 and not t.spool_path.exists()
    payload = {'a':[1]}
    a = t.log(task='task / with spaces', inputs=payload)
    payload['a'].append(2)
    b = t.log(task='second', inputs={}, outcome=None)
    rows = stored(t.spool_path)
    assert rows[0][0]['inputs'] == {'a':[1]}
    assert 'outcome' not in rows[0][0] and 'endpoint' not in rows[0][0] and rows[1][0]['outcome'] is None
    assert [r[0]['trace_id'] for r in rows] == [a,b]
    assert t.pending == 2
    with sqlite3.connect(t.spool_path) as db:
        assert db.execute('PRAGMA auto_vacuum').fetchone()[0] == 1
        assert db.execute('PRAGMA page_size').fetchone()[0] == 4096
    result = t.flush(timeout=0)
    assert result == TraceFlushResult(False,2,0)
    t.close(timeout=0); t.close(timeout=0)
    for action in [lambda:t.log(task='late',inputs={}), lambda:t.flush(), lambda:t.pending]:
        with pytest.raises(RuntimeError): action()
    assert result.pending == 2 and stored(t.spool_path)[0][0]['trace_id'] == a
    if os.name != 'nt': assert t.spool_path.stat().st_mode & 0o077 == 0


def test_delivery_ack_and_reopen_recovery(tmp_path):
    first = quiet(recorder(tmp_path)); identity = first.log(task='recover',inputs={})
    first.close(timeout=0)
    second = recorder(tmp_path); seen=[]
    second._send = lambda rows: (seen.extend(rows) or ack(rows))
    assert second.flush(timeout=2).complete
    assert [r['trace_id'] for r in seen] == [identity]
    assert second.pending == 0
    second.close(timeout=0)


def test_partial_rejection_quarantines_and_does_not_block_other_records(tmp_path):
    t = quiet(recorder(tmp_path))
    a=t.log(task='rejected',inputs={}); b=t.log(task='accepted',inputs={})
    del t._start
    def send(rows):
        return [r['trace_id'] for r in rows if r['trace_id'] != a], [dict(trace_id=a,code='trace_conflict',retryable=False)] if any(r['trace_id']==a for r in rows) else []
    t._send=send
    r=t.flush(timeout=2)
    assert not r.complete and r.pending==1 and r.rejected==1
    assert stored(t.spool_path)[0][1:] == ('rejected','trace_conflict')
    c=t.log(task='later',inputs={})
    limit=time.monotonic()+2
    while t.pending>1 and time.monotonic()<limit: time.sleep(.01)
    assert t.pending==1
    assert [r[0]['trace_id'] for r in stored(t.spool_path)]==[a]
    t.close(timeout=0)


def test_storage_warning_episodes_and_recovery(tmp_path, monkeypatch, caplog):
    t = quiet(recorder(tmp_path))
    original=t._db
    from contextlib import contextmanager
    @contextmanager
    def failed(*args,**kwargs): raise TraceStorageError('injected'); yield
    monkeypatch.setattr(t,'_db',failed)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(lambda _:t.log(task='fail',inputs={}),range(20))) == [None]*20
        assert len(seen)==1
        @contextmanager
        def full(*args,**kwargs): raise SpoolFullError('full'); yield
        monkeypatch.setattr(t,'_db',full)
        assert t.log(task='different',inputs={}) is None
        assert len(seen)==2
        monkeypatch.setattr(t,'_db',original)
        with caplog.at_level('INFO',logger='saina.traces'):
            assert t.log(task='recovered',inputs={})
        assert '1 records failed to persist' in caplog.text
        monkeypatch.setattr(t,'_db',failed)
        assert t.log(task='new episode',inputs={}) is None
        assert len(seen)==3
    monkeypatch.setattr(t,'_db',original);t.close(timeout=0)


def test_strict_capacity_and_lazy_unwritable_initialization(tmp_path):
    t=quiet(recorder(tmp_path,capacity=1024,on_storage_error='raise'))
    assert t.log(task='one',inputs={'text':'x'*400})
    with pytest.raises(SpoolFullError):t.log(task='two',inputs={'text':'x'*400})
    assert t.pending==1;t.close(timeout=0)
    blocker=tmp_path/'blocker';blocker.write_text('file')
    w=Traces(api_key='key',spool_path=blocker/'spool.db')
    with pytest.warns(RuntimeWarning): assert w.log(task='fail',inputs={}) is None
    # No successful admission exists here; an empty flush creates no storage.
    assert w.flush().complete
    w.close(timeout=0)
    with pytest.raises(RuntimeError):w.pending
    corrupt=tmp_path/'corrupt.db';corrupt.write_bytes(b'not sqlite')
    w=Traces(api_key='key',spool_path=corrupt)
    with pytest.raises(TraceStorageError):w.flush()
    with pytest.raises(TraceStorageError):w.close(timeout=0)


def test_binding_mismatch_never_mixes_credentials_and_paths_are_hashed(tmp_path,monkeypatch):
    first=quiet(recorder(tmp_path));first.log(task='one',inputs={});first.close(timeout=0)
    other=Traces('http://localhost:1',api_key='different',spool_path=tmp_path/'spool.db',on_storage_error='raise')
    with pytest.raises(TraceStorageError):other.log(task='two',inputs={})
    try:other.close(timeout=0)
    except TraceStorageError:pass
    monkeypatch.setenv('HOME',str(tmp_path))
    a=Traces(api_key='raw-secret');b=Traces(api_key='other-secret');c=Traces('https://other.example',api_key='raw-secret')
    assert 'raw-secret' not in str(a.spool_path) and len({a.spool_path,b.spool_path,c.spool_path})==3
    for t in [a,b,c]:t.close(timeout=0)


def test_admission_waits_out_a_peer_lock_longer_than_one_attempt(tmp_path):
    # A commit that outlasts the 250 ms SQLite busy timeout must not drop the record.
    path = tmp_path/'spool.db'
    seed = quiet(recorder(tmp_path))
    assert seed.log(task='seed', inputs={})
    seed.close(timeout=0)
    started = threading.Event()

    def hold():
        db = sqlite3.connect(path, timeout=5, isolation_level=None)
        db.execute('BEGIN IMMEDIATE')
        started.set()
        time.sleep(.6)
        db.commit()
        db.close()

    thread = threading.Thread(target=hold)
    thread.start()
    assert started.wait(2)
    writer = quiet(recorder(tmp_path))
    began = time.monotonic()
    try:
        assert writer.log(task='queued', inputs={})
        assert time.monotonic() - began >= .4
        assert len(stored(path)) == 2
    finally:
        writer.close(timeout=0)
        thread.join(3)


def test_crash_after_commit_and_concurrent_process_admission(tmp_path):
    code='''import os,sys
from saina import Traces
t=Traces('http://localhost:1',api_key='super-secret-key',spool_path=sys.argv[1])
t._start=lambda:None
for i in range(8): assert t.log(task='subprocess',inputs={'n':i})
os._exit(0)
'''
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    processes=[subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'spool.db')],env=env) for _ in range(3)]
    assert [p.wait(timeout=10) for p in processes]==[0]*3
    assert len(stored(tmp_path/'spool.db'))==24
    t=recorder(tmp_path);t._send=ack
    assert t.flush(timeout=3).complete;t.close(timeout=0)


def test_flush_snapshot_concurrent_admission_and_close_budget(tmp_path):
    t=quiet(recorder(tmp_path));a=t.log(task='first',inputs={})
    entered,release=threading.Event(),threading.Event()
    def send(rows):
        entered.set();release.wait(3);return ack(rows)
    t._send=send;del t._start
    with ThreadPoolExecutor(max_workers=2) as pool:
        future=pool.submit(t.flush,2)
        assert entered.wait(1)
        b=t.log(task='second',inputs={})
        release.set()
        result=future.result(timeout=3)
        assert result.complete
    t.close(timeout=0)
    assert result.pending>=0
    # A hung delivery cannot make close wait for the transport's full timeout.
    u=quiet(recorder(tmp_path/'another'));u.log(task='block',inputs={})
    entered.clear();release.clear();u._send=send;del u._start;u._start();assert entered.wait(1)
    started=time.monotonic();u.close(timeout=.05)
    assert time.monotonic()-started<.5
    release.set()
    if u._thread:u._thread.join(1)
    assert not u._thread.is_alive()


@pytest.mark.parametrize('kwargs', [dict(task='',inputs={}),dict(task='x',inputs={1:'bad'}),dict(task='x',inputs=float('nan')),dict(task='x',inputs=object()),dict(task='x',inputs='x'*(1<<20)),dict(task='x',inputs={},endpoint=1),dict(task='x',inputs={},endpoint=''),dict(task='x',inputs={},endpoint='  ')])
def test_invalid_arguments_never_create_storage(tmp_path,kwargs):
    t=recorder(tmp_path)
    with pytest.raises((TypeError,ValueError)):t.log(**kwargs)
    assert not t.spool_path.exists();t.close(timeout=0)


def test_optional_endpoint_is_sent_only_when_supplied(tmp_path):
    t = quiet(recorder(tmp_path))
    t.log(task='plain', inputs={})
    t.log(task='named', inputs={}, endpoint='support-triage')
    rows = stored(t.spool_path)
    assert 'endpoint' not in rows[0][0]
    assert rows[1][0]['endpoint'] == 'support-triage'
    assert set(rows[0][0]) == {'trace_id', 'recorded_at', 'task', 'inputs', 'metadata'}
    t.close(timeout=0)


def test_explicit_quarantine_export_retry_and_discard(tmp_path):
    t=quiet(recorder(tmp_path));a=t.log(task='reject',inputs={});b=t.log(task='pending',inputs={})
    with t._db() as db:db.execute("UPDATE records SET state='rejected',error='invalid_trace' WHERE trace_id=?",(a,))
    export=tmp_path/'recovery.jsonl'
    assert t.export_rejected(export)==1
    assert json.loads(export.read_text())['record']['trace_id']==a
    with pytest.raises(TraceStorageError):t.export_rejected(export)
    with pytest.raises(ValueError):t.discard_rejected([a,b])
    assert t.pending==2
    assert t.retry_rejected([a])==1
    with pytest.raises(ValueError):t.discard_rejected([a])
    with t._db() as db:db.execute("UPDATE records SET state='rejected',error='trace_conflict' WHERE trace_id=?",(a,))
    assert t.discard_rejected([a])==1 and t.pending==1
    with t._db() as db:
        used=int(db.execute("SELECT value FROM settings WHERE name='used'").fetchone()[0])
        assert used==db.execute('SELECT sum(size+256) FROM records').fetchone()[0]
    t.close(timeout=0)


def test_hanging_http_body_is_cancelled_on_close(tmp_path):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    entered,release=threading.Event(),threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Length','100');self.send_header('Connection','close');self.end_headers()
            self.wfile.write(b'{');self.wfile.flush();entered.set();release.wait(3)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    t=Traces(f'http://127.0.0.1:{server.server_port}',api_key='key',spool_path=tmp_path/'spool.db')
    try:
        t.log(task='blocked response',inputs={});assert entered.wait(1)
        before=time.monotonic();t.close(timeout=.05);assert time.monotonic()-before<.5
        t._thread.join(1);assert not t._thread.is_alive()
        assert len(stored(t.spool_path))==1
    finally:release.set();server.shutdown();server.server_close();thread.join(1)
