"""Durable external logging. Each SQLite connection belongs to its calling thread."""
import atexit
from collections.abc import Mapping
from functools import partial
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import socket
import sqlite3
import sys
import threading
import time
from urllib.parse import urlsplit
import warnings
import weakref

from .client import Saina, new_idempotency_key, _retry_after
from .errors import ProtocolError, SpoolFullError, TraceStorageError
from .validation import json_copy, json_value

_LOG = logging.getLogger('saina.traces')
_UNSET = object()
_MAX_RECORD = 1 << 20
_MAX_BATCH = 8 << 20
_APP_ID = 0x5341494E


@dataclass(frozen=True)
class TraceFlushResult:
    complete: bool
    pending: int
    rejected: int = 0


def _budget(value):
    if type(value) not in (int, float):
        raise TypeError('timeout must be a number')
    if not 0 <= value < float('inf'):
        raise ValueError('timeout must be nonnegative and finite')
    return value


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _depth(value, level=0):
    if level > 60:
        raise ValueError('Trace JSON nesting exceeds 60')
    if isinstance(value, dict):
        for item in value.values(): _depth(item, level + 1)
    elif isinstance(value, list):
        for item in value: _depth(item, level + 1)


def _path(base_url, key):
    u = urlsplit(base_url)
    origin = u._replace(scheme=u.scheme.lower(), netloc=u.netloc.lower(), path=u.path.rstrip('/')).geturl()
    service = hashlib.sha256(origin.encode()).hexdigest()
    credential = hashlib.sha256(key.encode()).hexdigest()
    binding = hashlib.sha256((service + ':' + credential).encode()).hexdigest()
    if sys.platform == 'win32':
        root = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'Saina'
    elif sys.platform == 'darwin':
        root = Path.home() / 'Library' / 'Application Support' / 'Saina'
    else:
        configured = os.environ.get('XDG_STATE_HOME')
        root = (Path(configured) if configured and Path(configured).is_absolute() else Path.home() / '.local' / 'state') / 'saina'
    return root / 'traces' / service / credential / 'spool.sqlite3', binding


def _shutdown(reference):
    recorder = reference()
    if recorder is not None:
        try: recorder.close(timeout=5)
        except Exception: pass


class Traces:
    """Commit logs locally before returning IDs; deliver them in a background worker.

    Capacity counts UTF-8 record bytes plus 256 bytes per row, including quarantine.
    Filesystems can temporarily need twice capacity plus 4 MiB for SQLite journals.
    Call flush() after restarting to drain an existing spool without logging anew.
    """
    def __init__(self, base_url=None, *, api_key=None, timeout=None, allow_http=False,
                 spool_path=None, capacity=64 << 20, on_storage_error='warn'):
        self._client = Saina(base_url, api_key=api_key, timeout=timeout, allow_http=allow_http)
        if on_storage_error not in ('warn', 'raise'):
            raise ValueError('on_storage_error must be warn or raise')
        if type(capacity) is not int:
            raise TypeError('capacity must be an integer')
        if capacity < 1024:
            raise ValueError('capacity must be at least 1024 bytes')
        if spool_path is not None and not isinstance(spool_path, (str, os.PathLike)):
            raise TypeError('spool_path must be a filesystem path')
        if spool_path is not None and not os.fspath(spool_path):
            raise ValueError('spool_path must not be empty')
        default, self._binding = _path(self._client.base_url, self._client._key)
        self.spool_path = Path(spool_path).expanduser().absolute() if spool_path is not None else default
        self.capacity, self.on_storage_error = capacity, on_storage_error
        self._condition = threading.Condition(threading.RLock())
        self._io = threading.RLock()
        self._wake, self._stop, self._closed_event = threading.Event(), threading.Event(), threading.Event()
        self._state, self._initialized, self._thread = 'open', False, None
        self._identity = None
        self._active, self._active_socket, self._force_retry = None, None, False
        self._episode, self._failures = None, 0
        self._attempts, self.last_error = 0, None
        # Register a weak reference, not a bound method retaining the recorder.
        self._exit = partial(_shutdown, weakref.ref(self))
        atexit.register(self._exit)

    def _check(self):
        if self._state != 'open':
            raise RuntimeError('Trace recorder is closed')

    @contextmanager
    def _db(self, create=True):
        with self._io:
            db = None
            try:
                if self._initialized:
                    stat = self.spool_path.stat()
                    if (stat.st_dev, stat.st_ino) != self._identity:
                        raise TraceStorageError('Spool file was replaced')
                if not self.spool_path.exists() and not create:
                    yield None
                    return
                if not self.spool_path.parent.exists():
                    self.spool_path.parent.mkdir(parents=True, mode=0o700)
                if not self.spool_path.exists():
                    try:
                        fd = os.open(self.spool_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    except FileExistsError:
                        pass
                    else:
                        os.close(fd)
                db = sqlite3.connect(self.spool_path, timeout=.25, isolation_level=None)
                db.execute('PRAGMA synchronous=FULL')
                if not self._initialized:
                    # auto_vacuum must be enabled outside a transaction, before any tables.
                    if (db.execute('PRAGMA application_id').fetchone()[0] in (0, _APP_ID) and
                            not db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()):
                        db.execute('PRAGMA auto_vacuum=FULL')
                    db.execute('BEGIN IMMEDIATE')
                    app = db.execute('PRAGMA application_id').fetchone()[0]
                    tables = db.execute("select name from sqlite_master where type='table'").fetchall()
                    if app not in (0, _APP_ID) or app == 0 and tables:
                        raise TraceStorageError('Path contains a different database')
                    if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
                        raise TraceStorageError('Spool must use SQLite DELETE journaling')
                    if db.execute('PRAGMA auto_vacuum').fetchone()[0] != 1:
                        raise TraceStorageError('Unsupported spool vacuum mode')
                    if db.execute('PRAGMA page_size').fetchone()[0] != 4096:
                        raise TraceStorageError('Unsupported spool page size')
                    db.execute(f'PRAGMA max_page_count={(self.capacity + (2 << 20)) // 4096}')
                    db.execute('CREATE TABLE IF NOT EXISTS settings(name TEXT PRIMARY KEY,value TEXT NOT NULL)')
                    db.execute("CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY AUTOINCREMENT,trace_id TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,size INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'pending',error TEXT)")
                    old = db.execute("SELECT value FROM settings WHERE name='binding'").fetchone()
                    if old is not None and old[0] != self._binding:
                        raise TraceStorageError('Spool belongs to a different service or credential')
                    db.execute("INSERT OR IGNORE INTO settings VALUES('binding',?)", (self._binding,))
                    old_capacity = db.execute("SELECT value FROM settings WHERE name='capacity'").fetchone()
                    if old_capacity is not None and int(old_capacity[0]) != self.capacity:
                        raise TraceStorageError('Spool capacity differs from its existing configuration')
                    db.execute("INSERT OR IGNORE INTO settings VALUES('capacity',?)", (str(self.capacity),))
                    db.execute("INSERT OR IGNORE INTO settings SELECT 'used',coalesce(sum(size+256),0) FROM records")
                    db.execute("CREATE TRIGGER IF NOT EXISTS record_admitted AFTER INSERT ON records BEGIN UPDATE settings SET value=cast(value AS INTEGER)+new.size+256 WHERE name='used'; END")
                    db.execute("CREATE TRIGGER IF NOT EXISTS record_removed AFTER DELETE ON records BEGIN UPDATE settings SET value=cast(value AS INTEGER)-old.size-256 WHERE name='used'; END")
                    db.execute('CREATE INDEX IF NOT EXISTS records_state_seq ON records(state,seq)')
                    db.execute(f'PRAGMA application_id={_APP_ID}')
                    db.commit()
                    stat = self.spool_path.stat()
                    self._identity = (stat.st_dev, stat.st_ino)
                    self._initialized = True
                # Every connection sets this limit; it is not a persistent PRAGMA.
                db.execute(f'PRAGMA max_page_count={(self.capacity + (2 << 20)) // 4096}')
                db.execute('BEGIN IMMEDIATE')
                yield db
                db.commit()
            except (OSError, sqlite3.Error) as exc:
                error = SpoolFullError('Trace spool is full') if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL else TraceStorageError('Trace storage is unavailable')
                error.category = (type(exc).__name__, getattr(exc, 'sqlite_errorcode', getattr(exc, 'errno', None)))
                raise error from None
            finally:
                if db is not None:
                    db.close()

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name='saina-traces', daemon=True)
            self._thread.start()
        self._wake.set()

    def _failed_log(self, error):
        if self.on_storage_error == 'raise':
            raise error
        category = (type(error).__name__, getattr(error, 'category', None))
        if self._episode != category:
            if self._episode is not None:
                _LOG.info('Trace storage failure category changed; %d records failed to persist', self._failures)
            self._episode, self._failures = category, 0
            try:
                warnings.warn('Trace storage failed; this record was not queued', RuntimeWarning, stacklevel=3)
            except Warning:
                pass
        self._failures += 1
        return None

    def log(self, *, task, inputs, outcome=_UNSET, metadata=None, endpoint=None):
        """Commit one schema 1.0 record. ``endpoint`` is omitted when unset."""
        with self._condition:
            self._check()
            if not isinstance(task, str): raise TypeError('task must be a string')
            if not task.strip() or len(task) > 1024: raise ValueError('task must be nonblank and at most 1024 characters')
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError('metadata must be a mapping')
            if endpoint is not None:
                if not isinstance(endpoint, str): raise TypeError('endpoint must be a string')
                if not endpoint.strip(): raise ValueError('endpoint must be a nonblank string')
            record = dict(trace_id=new_idempotency_key(), recorded_at=datetime.now(timezone.utc).isoformat(),
                          task=task, inputs=inputs, metadata={} if metadata is None else metadata)
            if outcome is not _UNSET: record['outcome'] = outcome
            if endpoint is not None: record['endpoint'] = endpoint
            json_value(record)
            record = json_copy(record)
            _depth(record)
            payload = _encode(record)
            size = len(payload.encode('utf-8'))
            if size > _MAX_RECORD: raise ValueError('Trace exceeds 1 MiB; use asset references')
            try:
                with self._db() as db:
                    used = int(db.execute("SELECT value FROM settings WHERE name='used'").fetchone()[0])
                    if used + size + 256 > self.capacity: raise SpoolFullError('Trace spool is full')
                    db.execute('INSERT INTO records(trace_id,payload,size) VALUES(?,?,?)', (record['trace_id'], payload, size))
            except (SpoolFullError, TraceStorageError) as error:
                return self._failed_log(error)
            if self._episode is not None:
                _LOG.info('Trace storage recovered; %d records failed to persist', self._failures)
                self._episode, self._failures = None, 0
            self._start()
            return record['trace_id']

    def _status(self, target=None):
        with self._db(create=False) as db:
            if db is None: return 0, 0, 0, 0
            total, rejected, last = db.execute("SELECT count(*),coalesce(sum(state='rejected'),0),coalesce(max(seq),0) FROM records").fetchone()
            remaining = total if target is None else db.execute('SELECT count(*) FROM records WHERE seq<=?', (target,)).fetchone()[0]
            return total, rejected, last, remaining

    @property
    def pending(self):
        with self._condition:
            self._check()
            total = self._status()[0]
            if total: self._start()
            return total

    def flush(self, timeout=5):
        with self._condition:
            self._check()
        return self._flush(time.monotonic() + _budget(timeout))

    def _flush(self, deadline):
        with self._condition:
            total, rejected, target, remaining = self._status()
            if not remaining: return TraceFlushResult(True, total, rejected)
            attempts = self._attempts
            self._force_retry = True
            self._start()
            self._condition.notify_all()
            while True:
                total, rejected, _, remaining = self._status(target)
                if not remaining: return TraceFlushResult(True, total, rejected)
                with self._db(create=False) as db:
                    deliverable = db.execute("SELECT count(*) FROM records WHERE seq<=? AND state='pending'", (target,)).fetchone()[0] if db else 0
                left = deadline - time.monotonic()
                if isinstance(self.last_error, (TraceStorageError, SpoolFullError)) and self._attempts > attempts:
                    raise self.last_error
                if self._state == 'closed' or left <= 0 or not deliverable or self.last_error is not None and self._attempts > attempts:
                    return TraceFlushResult(False, total, rejected)
                self._condition.wait(min(left, .1))

    def _batch(self):
        with self._db(create=False) as db:
            if db is None: return []
            rows = db.execute("SELECT trace_id,payload,size FROM records WHERE state='pending' ORDER BY seq LIMIT 100").fetchall()
        batch, size = [], 0
        for identity, payload, n in rows:
            if size + n > _MAX_BATCH: break
            batch.append(json.loads(payload)); size += n
        return batch

    def _send(self, rows):
        u = urlsplit(self._client.base_url)
        connection = (http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection)(u.hostname, u.port, timeout=min(5., self._client.timeout))
        with self._condition:
            if self._state == 'closed': return None
            self._active = connection
        try:
            connection.request('POST', u.path.rstrip('/') + '/v1/traces',
                               body=_encode({'schema_version': '1.0', 'traces': rows}).encode(),
                               headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._client._key})
            with self._condition:
                if self._state == 'closed': return None
                self._active_socket = connection.sock
            response = connection.getresponse()
            raw = response.read((2 << 20) + 1)
            if not 200 <= response.status < 300 or len(raw) > 2 << 20:
                raise ProtocolError('Trace delivery failed', response.status, retry_after=_retry_after(response.headers))
            receipt = json.loads(raw)
            expected = {r['trace_id'] for r in rows}
            accepted, rejected = receipt.get('accepted'), receipt.get('rejected')
            if receipt.get('schema_version') != '1.0' or not isinstance(accepted, list) or not isinstance(rejected, list):
                raise ProtocolError('Invalid trace acknowledgment')
            ids = list(accepted)
            for item in rejected:
                if (not isinstance(item, dict) or item.get('retryable') is not False or
                        item.get('code') not in ('invalid_trace', 'trace_conflict')):
                    raise ProtocolError('Invalid trace rejection')
                ids.append(item.get('trace_id'))
            if any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or set(ids) != expected:
                raise ProtocolError('Invalid trace acknowledgment IDs')
            return accepted, rejected
        finally:
            connection.close()
            with self._condition: self._active, self._active_socket = None, None

    def _worker(self):
        delay = .1
        while not self._stop.is_set():
            try:
                rows = self._batch()
                if not rows:
                    self._wake.wait(.5); self._wake.clear(); continue
                receipt = self._send(rows)
                with self._condition:
                    if self._state == 'closed': break
                    if receipt is not None:
                        accepted, rejected = receipt
                        with self._db() as db:
                            db.executemany('DELETE FROM records WHERE trace_id=?', [(i,) for i in accepted])
                            db.executemany("UPDATE records SET state='rejected',error=? WHERE trace_id=?",
                                           [(i['code'], i['trace_id']) for i in rejected])
                    self.last_error = None
                    self._attempts += 1
                    self._condition.notify_all()
                delay = .1
            except Exception as error:
                with self._condition:
                    self.last_error = error if isinstance(error, (TraceStorageError, SpoolFullError)) else ProtocolError('Trace delivery failed; records remain queued', getattr(error, 'status', None), retry_after=getattr(error, 'retry_after', None))
                    self._attempts += 1
                    self._condition.notify_all()
                with self._condition:
                    self._condition.wait_for(lambda: self._stop.is_set() or self._force_retry,
                                             timeout=max(delay, min(86400, getattr(error, 'retry_after', None) or 0)))
                    self._force_retry = False
                delay = min(30., delay * 2)

    def export_rejected(self, path):
        """Write a new private JSONL recovery file; never delete spool records."""
        with self._condition:
            self._check()
            with self._db(create=False) as db:
                rows = db.execute("SELECT payload,error FROM records WHERE state='rejected' ORDER BY seq").fetchall() if db else []
            try:
                fd = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    for payload, error in rows:
                        stream.write(_encode({'record': json.loads(payload), 'reason': error}) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
            except OSError:
                raise TraceStorageError('Could not create trace recovery export') from None
            return len(rows)

    def _recover(self, trace_ids, retry):
        with self._condition:
            self._check()
            if not isinstance(trace_ids, (list, tuple)) or any(not isinstance(i, str) for i in trace_ids):
                raise TypeError('trace_ids must be a list of strings')
            if len(set(trace_ids)) != len(trace_ids): raise ValueError('trace_ids must be unique')
            if not trace_ids: return 0
            with self._db(create=False) as db:
                if db is None: raise ValueError('Unknown rejected trace ID')
                for identity in trace_ids:
                    row = db.execute('SELECT state FROM records WHERE trace_id=?', (identity,)).fetchone()
                    if row is None or row[0] != 'rejected': raise ValueError('Only quarantined trace IDs may be recovered')
                statement = "UPDATE records SET state='pending',error=NULL WHERE trace_id=?" if retry else 'DELETE FROM records WHERE trace_id=?'
                db.executemany(statement, [(i,) for i in trace_ids])
            if retry: self._start()
            return len(trace_ids)

    def retry_rejected(self, trace_ids):
        """Retry unchanged quarantined records after fixing the rejection cause."""
        return self._recover(trace_ids, True)

    def discard_rejected(self, trace_ids):
        """Explicitly remove ONLY quarantined records, after operator export/review."""
        return self._recover(trace_ids, False)

    def close(self, timeout=5):
        deadline = time.monotonic() + _budget(timeout)
        with self._condition:
            if self._state == 'closed': return
            if self._state == 'closing':
                owner = False
            else:
                self._state, owner = 'closing', True
        if not owner:
            self._closed_event.wait(max(0, deadline - time.monotonic()))
            return
        try:
            self._flush(deadline)
        finally:
            with self._condition:
                self._state = 'closed'
                self._stop.set(); self._wake.set()
                active = self._active
                if active is not None:
                    try:
                        if self._active_socket is not None: self._active_socket.shutdown(socket.SHUT_RDWR)
                        elif active.sock is not None: active.sock.shutdown(socket.SHUT_RDWR)
                    except OSError: pass
                    active.close()
                self._condition.notify_all()
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(max(0, deadline - time.monotonic()))
            self._closed_event.set()
            atexit.unregister(self._exit)

    def __enter__(self):
        with self._condition: self._check()
        return self

    def __exit__(self, kind, error, tb):
        try: self.close()
        except (TraceStorageError, SpoolFullError):
            if error is None: raise
        return False


from .legacy_traces import TraceClient, DecisionTrace, TraceDeliveryError
