"""Explicit decision evidence capture. This module never invokes a model implicitly."""
from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
import warnings
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


class TraceDeliveryError(RuntimeError):
    """Trace delivery failed. Queued evidence remains available for retry."""
    def __init__(self, message, *, status=None, code=None):
        super().__init__(message)
        self.status, self.code = status, code


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', value):
        raise ValueError('Use a stable identifier of 1–120 letters, digits, dots, underscores or hyphens')
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DecisionTrace:
    """A single-use sync/async scope; only explicit inputs and answers are captured."""
    def __init__(self, client, task, inputs, *, decision_id=None, metadata=None):
        self.client = client
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError('Trace metadata must be an object')
        self.decision_id = _id(decision_id or 'trace_' + uuid4().hex)
        self._record = {'schema_version': '1.0', 'decision_id': self.decision_id,
                        'task': _id(task), 'inputs': _snapshot(inputs),
                        'metadata': _snapshot(metadata or {}), 'recorded_at': _now()}
        self._entered = self._finished = self._has_answer = False

    def set_answer(self, answer):
        if self._finished:
            raise RuntimeError('Decision scope is already finished')
        self._record['answer'] = _snapshot(answer)
        self._has_answer = True

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Decision scopes cannot be reused')
        self.client._check_open()
        self._entered = True
        self._record['recorded_at'] = _now()
        self._started = time.perf_counter()
        return self

    def __exit__(self, kind, error, tb):
        self._finished = True
        self._record.update(completed_at=_now(), timing={'total_ms': (time.perf_counter()-self._started)*1000},
                            status='error' if error is not None else 'completed' if self._has_answer else 'incomplete')
        if error is not None:
            # Exception messages/stack traces can contain credentials and arbitrary payloads.
            self._record['error'] = {'type': kind.__name__}
        try:
            self.client._enqueue(self._record)
        except Exception as exc:
            self.client._notify(exc)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, kind, error, tb):
        return self.__exit__(kind, error, tb)


class TraceClient:
    """Trace-only collection client; constructor and decision scopes do no network I/O.

    Background batches use memory only. Call flush/close (aflush/aclose in async code)
    to await durable acknowledgement before shutdown. Delivery never retries your
    decision function. Use invoke_endpoint explicitly to invoke a Saina endpoint.
    """
    def __init__(self, *, collection, api_key=None, base_url='https://api.saina.run',
                 source='application', batch_size=100, flush_interval=5.0,
                 max_pending=1000, timeout=10.0, max_retries=2, on_error=None):
        self.collection, self.source = _id(collection), _id(source)
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment or url.path.rstrip('/'):
            raise ValueError('Use an explicit HTTP(S) service origin')
        if api_key is not None and (not isinstance(api_key, str) or not api_key or '\r' in api_key or '\n' in api_key):
            raise ValueError('Invalid API key')
        if type(batch_size) is not int or not 1 <= batch_size <= 100:
            raise ValueError('batch_size must be 1–100')
        if type(max_pending) is not int or max_pending < batch_size:
            raise ValueError('max_pending must be at least batch_size')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (flush_interval, timeout)):
            raise ValueError('flush_interval and timeout must be positive finite numbers')
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError('max_retries must be 0–5')
        if on_error is not None and not callable(on_error):
            raise ValueError('on_error must be callable')
        self.base_url, self._api_key = base_url.rstrip('/'), api_key
        self.batch_size, self.flush_interval, self.max_pending = batch_size, flush_interval, max_pending
        self.timeout, self.max_retries, self.on_error = timeout, max_retries, on_error
        self._lock, self._send_lock = threading.Lock(), threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._queue, self._thread = [], None
        self._closed, self._closing = False, False
        self.last_error = None
        self.dropped_count = 0

    def _check_open(self):
        if self._closed or self._closing:
            raise RuntimeError('Trace client is closed')

    @property
    def pending_count(self):
        with self._lock:
            return len(self._queue)

    def _notify(self, error):
        self.last_error = error
        try:
            if self.on_error:
                self.on_error(error)
            else:
                warnings.warn('Saina trace capture/delivery failed; inspect last_error and pending_count', RuntimeWarning, stacklevel=2)
        except Exception:
            pass  # A telemetry error callback must never replace an application exception.

    def decision(self, *, task, inputs, decision_id=None, metadata=None):
        self._check_open()
        return DecisionTrace(self, task, inputs, decision_id=decision_id, metadata=metadata)

    def _enqueue(self, record):
        record = _snapshot(record)
        if len(json.dumps(record, ensure_ascii=False).encode()) > 1048576:
            self.dropped_count += 1
            raise TraceDeliveryError('Trace exceeds 1 MiB; use asset references')
        with self._lock:
            self._check_open()
            if len(self._queue) >= self.max_pending:
                self.dropped_count += 1
                raise TraceDeliveryError('Trace queue is full')
            self._queue.append(record)
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name='saina-traces', daemon=True)
                self._thread.start()
            if len(self._queue) >= self.batch_size:
                self._wake.set()

    def _worker(self):
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.flush()
            except Exception as exc:
                self._notify(exc)

    def _post(self, path, body, *, retry=True):
        data = json.dumps(body, allow_nan=False, ensure_ascii=False).encode()
        headers = {'Content-Type': 'application/json', 'X-Saina-Client': 'python-sdk'}
        if self._api_key:
            headers['Authorization'] = 'Bearer ' + self._api_key
        for attempt in range((self.max_retries if retry else 0)+1):
            delay = min(2**attempt * .25, 5)
            try:
                request = Request(self.base_url + path, data=data, headers=headers, method='POST')
                with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                    raw = response.read(2*1048576+1)
                    if len(raw) > 2*1048576:
                        raise TraceDeliveryError('Trace service response exceeds 2 MiB')
                    return json.loads(raw)
            except HTTPError as exc:
                code = None
                try:
                    code = json.loads(exc.read(8192)).get('error', {}).get('code')
                except (ValueError, AttributeError):
                    pass
                error = TraceDeliveryError('Saina trace request failed', status=exc.code, code=code)
                transient = exc.code in (408, 429, 500, 502, 503, 504)
                try:
                    delay = min(30, max(0, float(exc.headers.get('Retry-After', delay))))
                except (ValueError, TypeError):
                    pass
            except (OSError, URLError, ValueError):
                error = TraceDeliveryError('Saina trace connection or response failed')
                transient = True
            if not retry or not transient or attempt == self.max_retries:
                raise error from None
            time.sleep(delay)

    def flush(self):
        """Deliver queued evidence. A failed batch stays queued with its original IDs."""
        try:
            self._flush()
        except Exception as exc:
            self.last_error = exc
            raise

    def _flush(self):
        with self._send_lock:
            with self._lock:
                remaining = len(self._queue)
            while remaining:
                with self._lock:
                    # Bound by bytes as well as record count (hosted request limit: 25 MiB).
                    batch, size = [], 0
                    for record in self._queue[:min(self.batch_size, remaining)]:
                        n = len(json.dumps(record, ensure_ascii=False).encode())
                        if batch and size+n > 8*1048576:
                            break
                        batch.append(record)
                        size += n
                receipt = self._post('/v1/studio/data/collections/'+quote(self.collection, safe='')+'/traces',
                                     {'source': self.source, 'traces': batch})
                expected = [r['decision_id'] for r in batch]
                if receipt.get('decision_ids') != expected:
                    raise TraceDeliveryError('Trace service returned an invalid acknowledgement')
                with self._lock:
                    del self._queue[:len(batch)]
                remaining -= len(batch)
            self.last_error = None

    async def aflush(self):
        await asyncio.to_thread(self.flush)

    def close(self):
        self._closing = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self.flush()
        self._closed = True

    async def aclose(self):
        await asyncio.to_thread(self.close)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, kind, error, tb):
        try:
            self.close()
        except Exception as exc:
            if error is None:
                raise
            self._notify(exc)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, kind, error, tb):
        try:
            await self.aclose()
        except Exception as exc:
            if error is None:
                raise
            self._notify(exc)
        return False

    def report_outcome(self, decision_id, *, values, event_id):
        self._check_open()
        self.flush()
        return self._post('/v1/studio/data/collections/'+quote(self.collection, safe='')+'/traces/'+quote(_id(decision_id), safe='')+'/outcomes',
                          {'source': self.source, 'event_id': _id(event_id), 'values': _snapshot(values)})

    async def areport_outcome(self, decision_id, *, values, event_id):
        return await asyncio.to_thread(self.report_outcome, decision_id, values=values, event_id=event_id)

    def invoke_endpoint(self, endpoint, request, *, version='production'):
        """Explicit inference. Server captures exactly once; invocation is never retried."""
        self._check_open()
        if version != 'production' and (type(version) is not int or version < 1):
            raise ValueError('Choose production or a positive version')
        if not isinstance(request, dict) or any(k in request for k in ('version', 'trace_collection_id')):
            raise ValueError('Pass an endpoint request without version or trace_collection_id')
        return self._post('/v1/studio/endpoints/'+quote(_id(endpoint), safe='')+'/decide',
                          {**_snapshot(request), 'version': version, 'trace_collection_id': self.collection}, retry=False)

    async def ainvoke_endpoint(self, endpoint, request, *, version='production'):
        return await asyncio.to_thread(self.invoke_endpoint, endpoint, request, version=version)
