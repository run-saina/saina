/** Node-only durable logging; Node modules are loaded lazily so browser inference still imports. */
import {Saina, ProtocolError, SpoolFullError, TraceStorageError, newIdempotencyKey} from './index.js';
import {jsonValue} from './validation.js';
const MAX_RECORD = 1 << 20, MAX_BATCH = 8 << 20, APP_ID = 0x5341494e;
const unset = Symbol('unset');
const budget = value => {
  if (typeof value !== 'number') throw new TypeError('timeout must be a number');
  if (!Number.isFinite(value) || value < 0) throw new RangeError('timeout must be nonnegative and finite');
  return value * 1000;
};
const depth = (value, n = 0) => {
  if (typeof value === 'string' && /[\uD800-\uDFFF]/u.test(value)) throw new TypeError('Trace strings must be valid Unicode');
  if (n > 60) throw new RangeError('Trace JSON nesting exceeds 60');
  if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) { depth(key, n + 1); depth(item, n + 1); }
};
let dependencies;
const loadNode = specifier => import(/* webpackIgnore: true */ /* @vite-ignore */ specifier);
async function nodeModules() {
  dependencies ??= Promise.all(['node:sqlite', 'node:fs', 'node:path', 'node:os', 'node:crypto'].map(loadNode));
  const [sqlite, fs, path, os, crypto] = await dependencies;
  return {sqlite, fs, path, os, crypto};
}
const storageError = error => {
  if (error instanceof TraceStorageError || error instanceof SpoolFullError) return error;
  const result = error?.errcode === 13 ? new SpoolFullError('Trace spool is full') : new TraceStorageError('Trace storage is unavailable');
  result.category = `${error?.code ?? error?.name}:${error?.errcode ?? ''}`;
  return result;
};

export class Traces {
  #client; #fileIdentity; #fs; #db; #initPromise; #state = 'open'; #timer; #pumpPromise; #controller; #closePromise;
  #targets = new Map(); #closedTargets = new Map(); #closeStorageError; #spoolPath; #logs = new Set(); #waiters = new Set(); #attempts = 0; #delay = 100; #episode; #failures = 0; #exit;
  constructor({baseUrl, apiKey, timeout, allowHttp = false, spoolPath, capacity = 64 << 20,
    onStorageError = 'warn', fetch: fetchImpl} = {}) {
    const version = globalThis.process?.versions?.node?.split('.').map(Number);
    if (!version || version[0] < 22 || version[0] === 22 && version[1] < 13)
      throw new TypeError('Durable Traces requires Node.js 22.13 or newer');
    if (!['warn', 'raise'].includes(onStorageError)) throw new TypeError('onStorageError must be warn or raise');
    if (!Number.isSafeInteger(capacity) || capacity < 1024) throw new TypeError('capacity must be an integer of at least 1024 bytes');
    if (spoolPath !== undefined && (typeof spoolPath !== 'string' || !spoolPath)) throw new TypeError('spoolPath must be a nonempty string');
    this.#client = new Saina({baseUrl, apiKey, timeout, allowHttp, ...(fetchImpl ? {fetch: fetchImpl} : {})});
    this.#spoolPath = spoolPath; Object.defineProperties(this, {capacity: {value: capacity, enumerable: true}, onStorageError: {value: onStorageError}}); this.lastError = null;
    const reference = new WeakRef(this);
    this.#exit = () => reference.deref()?.close({timeout: 5}).catch(() => {});
    process.on('beforeExit', this.#exit);
  }
  get spoolPath() { return this.#spoolPath; }
  #check() { if (this.#state !== 'open') throw new Error('Trace recorder is closed'); }
  async #initialize(create) {
    const {sqlite, fs, path, os, crypto} = await nodeModules();
    this.#fs = fs;
    const hash = value => crypto.createHash('sha256').update(value).digest('hex');
    const parts = this.#client.baseUrl.match(/^([^:]+):\/\/([^/]+)(.*)$/);
    const service = hash(`${parts[1].toLowerCase()}://${parts[2].toLowerCase()}${parts[3].replace(/\/+$/, '')}`);
    const credential = hash(this.#client.apiKey), binding = hash(`${service}:${credential}`);
    if (this.spoolPath === undefined) {
      const home = os.homedir();
      const root = process.platform === 'win32' ? path.join(process.env.LOCALAPPDATA ?? path.join(home, 'AppData', 'Local'), 'Saina')
        : process.platform === 'darwin' ? path.join(home, 'Library', 'Application Support', 'Saina')
        : path.join(process.env.XDG_STATE_HOME && path.isAbsolute(process.env.XDG_STATE_HOME) ? process.env.XDG_STATE_HOME : path.join(home, '.local', 'state'), 'saina');
      this.#spoolPath = path.join(root, 'traces', service, credential, 'spool.sqlite3');
    } else this.#spoolPath = path.resolve(this.spoolPath);
    if (!create && !fs.existsSync(this.spoolPath)) return null;
    if (this.#state === 'closed') throw new Error('Trace recorder is closed');
    fs.mkdirSync(path.dirname(this.spoolPath), {recursive: true, mode: 0o700});
    try { const fd = fs.openSync(this.spoolPath, 'wx', 0o600); fs.closeSync(fd); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    const db = new sqlite.DatabaseSync(this.spoolPath);
    try {
      db.exec('PRAGMA busy_timeout=250; PRAGMA synchronous=FULL;');
      if ([0, APP_ID].includes(db.prepare('PRAGMA application_id').get().application_id) && !db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().length) db.exec('PRAGMA auto_vacuum=FULL');
      db.exec('BEGIN IMMEDIATE');
      const app = db.prepare('PRAGMA application_id').get().application_id;
      const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all();
      if (![0, APP_ID].includes(app) || app === 0 && tables.length) throw new TraceStorageError('Path contains a different database');
      if (db.prepare('PRAGMA journal_mode').get().journal_mode !== 'delete') throw new TraceStorageError('Spool must use SQLite DELETE journaling');
      if (db.prepare('PRAGMA auto_vacuum').get().auto_vacuum !== 1) throw new TraceStorageError('Unsupported spool vacuum mode');
      if (db.prepare('PRAGMA page_size').get().page_size !== 4096) throw new TraceStorageError('Unsupported spool page size');
      db.exec(`PRAGMA max_page_count=${Math.floor((this.capacity + (2 << 20))/4096)}`);
      db.exec("CREATE TABLE IF NOT EXISTS settings(name TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY AUTOINCREMENT,trace_id TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,size INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'pending',error TEXT);");
      const old = db.prepare("SELECT value FROM settings WHERE name='binding'").get();
      if (old && old.value !== binding) throw new TraceStorageError('Spool belongs to a different service or credential');
      const capacity = db.prepare("SELECT value FROM settings WHERE name='capacity'").get();
      if (capacity && Number(capacity.value) !== this.capacity) throw new TraceStorageError('Spool capacity differs from its existing configuration');
      db.prepare("INSERT OR IGNORE INTO settings VALUES('binding',?)").run(binding);
      db.prepare("INSERT OR IGNORE INTO settings VALUES('capacity',?)").run(String(this.capacity));
      db.exec("INSERT OR IGNORE INTO settings SELECT 'used',coalesce(sum(size+256),0) FROM records; CREATE TRIGGER IF NOT EXISTS record_admitted AFTER INSERT ON records BEGIN UPDATE settings SET value=cast(value AS INTEGER)+new.size+256 WHERE name='used'; END; CREATE TRIGGER IF NOT EXISTS record_removed AFTER DELETE ON records BEGIN UPDATE settings SET value=cast(value AS INTEGER)-old.size-256 WHERE name='used'; END; CREATE INDEX IF NOT EXISTS records_state_seq ON records(state,seq);");
      db.exec(`PRAGMA application_id=${APP_ID}; COMMIT;`);
      if (this.#state === 'closed') { db.close(); return null; }
      const stat = fs.statSync(this.spoolPath, {bigint: true}); this.#fileIdentity = `${stat.dev}:${stat.ino}`;
      this.#db = db;
      return db;
    } catch (error) { db.close(); throw error; }
  }
  async #ensure(create = false) {
    if (this.#db) return this.#db;
    const initialization = this.#initPromise ??= this.#initialize(create);
    let db;
    try { db = await initialization; }
    catch (error) { throw storageError(error); }
    finally { if (this.#initPromise === initialization) this.#initPromise = null; }
    // A read-only first use may find no file. Admissions sharing that operation
    // must upgrade to creation, without clearing a newer initialization promise.
    if (!db && create && this.#state !== 'closed') return this.#ensure(true);
    return db;
  }
  #transaction(callback) {
    try {
      const stat = this.#fs.statSync(this.spoolPath, {bigint: true});
      if (`${stat.dev}:${stat.ino}` !== this.#fileIdentity) throw new TraceStorageError('Spool file was replaced');
      this.#db.exec('BEGIN IMMEDIATE');
      try { const result = callback(this.#db); this.#db.exec('COMMIT'); return result; }
      catch (error) { this.#db.exec('ROLLBACK'); throw error; }
    } catch (error) { if (error instanceof RangeError || error instanceof TypeError) throw error; throw storageError(error); }
  }
  log(record = {}) {
    this.#check();
    if (!record || typeof record !== 'object' || Array.isArray(record) || Object.keys(record).some(k => !['task', 'inputs', 'outcome', 'metadata'].includes(k))) throw new TypeError('Invalid trace arguments');
    const {task, inputs, metadata = null} = record;
    const outcome = Object.hasOwn(record, 'outcome') ? record.outcome : unset;
    if (Object.hasOwn(record, 'metadata') && record.metadata === undefined) throw new TypeError('metadata must be JSON data');
    // Argument snapshots happen before the first asynchronous storage operation.
    if (typeof task !== 'string') throw new TypeError('task must be a string');
    if (!task.trim() || [...task].length > 1024) throw new RangeError('task must be nonblank and at most 1024 characters');
    if (metadata !== null && (typeof metadata !== 'object' || Array.isArray(metadata))) throw new TypeError('metadata must be an object');
    const row = {trace_id: newIdempotencyKey(), recorded_at: new Date().toISOString(), task, inputs, metadata: metadata ?? {}};
    if (outcome !== unset) row.outcome = outcome;
    jsonValue(row); depth(row);
    const payload = JSON.stringify(row), size = new TextEncoder().encode(payload).length;
    if (size > MAX_RECORD) throw new RangeError('Trace exceeds 1 MiB; use asset references');
    const operation = this.#log(row.trace_id, payload, size);
    this.#logs.add(operation);
    operation.finally(() => this.#logs.delete(operation)).catch(() => {});
    return operation;
  }
  async #log(id, payload, size) {
    try {
      await this.#ensure(true);
      if (this.#state === 'closed') throw new Error('Trace recorder is closed');
      this.#transaction(db => {
        const used = Number(db.prepare("SELECT value FROM settings WHERE name='used'").get().value);
        if (used + size + 256 > this.capacity) throw new SpoolFullError('Trace spool is full');
        db.prepare('INSERT INTO records(trace_id,payload,size) VALUES(?,?,?)').run(id, payload, size);
      });
    } catch (caught) {
      if (this.#state === 'closed') throw new Error('Trace recorder is closed');
      const error = storageError(caught);
      if (this.onStorageError === 'raise') throw error;
      const category = `${error.constructor.name}:${error.category ?? ''}`;
      if (category !== this.#episode) {
        if (this.#episode) console.info(`Trace storage failure category changed; ${this.#failures} records failed to persist`);
        this.#episode = category; this.#failures = 0;
        process.emitWarning('Trace storage failed; this record was not queued', {code: 'SAINA_TRACE_STORAGE'});
      }
      this.#failures++;
      return null;
    }
    if (this.#episode) {
      console.info(`Trace storage recovered; ${this.#failures} records failed to persist`);
      this.#episode = undefined; this.#failures = 0;
    }
    this.#kick(); return id;
  }
  #status(target = Infinity) {
    if (!this.#db) return {pending: 0, rejected: 0, last: 0, remaining: 0, deliverable: 0};
    return this.#transaction(db => {
      const r = db.prepare("SELECT count(*) AS pending,coalesce(sum(state='rejected'),0) AS rejected,coalesce(max(seq),0) AS last FROM records").get();
      const limit = Number.isFinite(target) ? target : r.last;
      const t = db.prepare("SELECT count(*) AS remaining,coalesce(sum(state='pending'),0) AS deliverable FROM records WHERE seq<=?").get(limit);
      return {...r, ...t};
    });
  }
  get pending() {
    this.#check();
    return this.#ensure(false).then(() => { this.#check(); const n = this.#status().pending; if (n) this.#kick(); return n; });
  }
  #notify() { for (const resolve of this.#waiters) resolve(); this.#waiters.clear(); }
  #kick(force = false) {
    if (this.#state === 'closed' || this.#pumpPromise) return;
    if (this.#timer && !force) return;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      const operation = this.#pump(); this.#pumpPromise = operation;
      operation.finally(() => { this.#pumpPromise = undefined; this.#notify(); }).catch(() => {});
    }, 0);
    this.#timer.unref();
  }
  async #send(rows) {
    const controller = new AbortController(); this.#controller = controller;
    try {
      const response = await this.#client.fetch(this.#client.baseUrl + '/v1/traces', {
        method: 'POST', redirect: 'error', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(Math.min(5000, this.#client.timeout))]),
        headers: {'Content-Type': 'application/json', Authorization: `Bearer ${this.#client.apiKey}`},
        body: '{"schema_version":"1.0","traces":[' + rows.map(row => row.payload).join(',') + ']}'});
      const chunks = []; let size = 0;
      const reader = response.body?.getReader();
      if (reader) {
        for (;;) {
          const {done, value} = await reader.read(); if (done) break;
          size += value.length;
          if (size > 2 << 20) { await reader.cancel(); throw new ProtocolError('Trace response is too large'); }
          chunks.push(value);
        }
      }
      if (!response.ok) {
        const after = response.headers.get('Retry-After');
        const delay = after === null ? 0 : /^\d+$/.test(after) ? Number(after)*1000 : Math.max(0, Date.parse(after)-Date.now());
        throw new ProtocolError('Trace delivery failed', response.status, {retryAfter: Number.isFinite(delay) ? Math.min(86400000, delay)/1000 : 0});
      }
      const buffer = new Uint8Array(size); let offset = 0;
      for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length; }
      const receipt = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(buffer));
      if (receipt.schema_version !== '1.0' || !Array.isArray(receipt.accepted) || !Array.isArray(receipt.rejected)) throw new ProtocolError('Invalid trace acknowledgment');
      const ids = [...receipt.accepted];
      for (const item of receipt.rejected) {
        if (!item || item.retryable !== false || !['invalid_trace', 'trace_conflict'].includes(item.code)) throw new ProtocolError('Invalid trace rejection');
        ids.push(item.trace_id);
      }
      const expected = new Set(rows.map(r => r.trace_id));
      if (ids.some(id => typeof id !== 'string' || !expected.has(id)) || ids.length !== expected.size || new Set(ids).size !== ids.length) throw new ProtocolError('Invalid trace acknowledgment IDs');
      return receipt;
    } finally { this.#controller = undefined; }
  }
  async #pump() {
    let attempted = false;
    try {
      const rows = this.#transaction(db => db.prepare("SELECT trace_id,payload,size FROM records WHERE state='pending' ORDER BY seq LIMIT 100").all());
      const batch = []; let size = 0;
      for (const row of rows) { if (size + row.size > MAX_BATCH) break; batch.push(row); size += row.size; }
      if (!batch.length) return;
      attempted = true;
      const receipt = await this.#send(batch);
      if (this.#state === 'closed') return;
      this.#transaction(db => {
        for (const id of receipt.accepted) db.prepare('DELETE FROM records WHERE trace_id=?').run(id);
        for (const item of receipt.rejected) db.prepare("UPDATE records SET state='rejected',error=? WHERE trace_id=?").run(item.code, item.trace_id);
      });
      this.lastError = null; this.#delay = 100;
    } catch (error) {
      this.lastError = error instanceof TraceStorageError || error instanceof SpoolFullError ? error : new ProtocolError('Trace delivery failed; records remain queued', error?.status, {retryAfter: error?.retryAfter});
    } finally {
      this.#attempts++; this.#notify();
      if (this.#state !== 'closed' && (attempted || this.lastError)) {
        const wait = this.lastError ? Math.max(this.#delay, this.lastError.retryAfter * 1000 || 0) : 0;
        if (this.lastError) this.#delay = Math.min(30000, this.#delay * 2);
        this.#timer = setTimeout(() => { this.#timer = undefined; this.#kick(); }, wait || 10);
        this.#timer.unref();
      }
    }
  }
  flush({timeout = 5} = {}) { this.#check(); return this.#flush(Date.now() + budget(timeout)); }
  async #flush(deadline) {
    await this.#ensure(false);
    if (this.#state === 'closed') throw new Error('Recorder closed before a flush snapshot was available');
    const initial = this.#status(), target = initial.last, attempts = this.#attempts;
    if (!initial.remaining) return Object.freeze({complete: true, pending: initial.pending, rejected: initial.rejected});
    this.#targets.set(target, (this.#targets.get(target) ?? 0) + 1);
    try {
      this.#kick(true);
      let status = initial;
      while (this.#state !== 'closed') {
        status = this.#status(target);
        if (!status.remaining) return Object.freeze({complete: true, pending: status.pending, rejected: status.rejected});
        if (this.#attempts > attempts && (this.lastError instanceof TraceStorageError || this.lastError instanceof SpoolFullError)) throw this.lastError;
        if (Date.now() >= deadline || !status.deliverable || this.#attempts > attempts && this.lastError) break;
        await new Promise(resolve => {
          const done = () => { clearTimeout(timer); this.#waiters.delete(done); resolve(); };
          const timer = setTimeout(done, Math.max(0, deadline - Date.now())); this.#waiters.add(done);
        });
      }
      if (this.#state === 'closed') {
        if (this.#closeStorageError) throw this.#closeStorageError;
        status = this.#closedTargets.get(target) ?? status;
      }
      return Object.freeze({complete: status.remaining === 0, pending: status.pending, rejected: status.rejected});
    } finally {
      const count = this.#targets.get(target) - 1;
      if (count) this.#targets.set(target, count);
      else { this.#targets.delete(target); this.#closedTargets.delete(target); }
    }
  }
  async exportRejected(path) {
    this.#check(); await this.#ensure(false); this.#check();
    const rows = this.#db ? this.#transaction(db => db.prepare("SELECT payload,error FROM records WHERE state='rejected' ORDER BY seq").all()) : [];
    const {fs} = await nodeModules(); this.#check();
    let fd;
    try {
      fd = fs.openSync(path, 'wx', 0o600);
      for (const row of rows) fs.writeFileSync(fd, '{"record":' + row.payload + ',"reason":' + JSON.stringify(row.error) + '}\n');
      fs.fsyncSync(fd);
    } catch { throw new TraceStorageError('Could not create trace recovery export'); }
    finally { if (fd !== undefined) fs.closeSync(fd); }
    return rows.length;
  }
  async #recover(ids, retry) {
    this.#check();
    if (!Array.isArray(ids) || ids.some(id => typeof id !== 'string')) throw new TypeError('traceIds must be an array of strings');
    if (new Set(ids).size !== ids.length) throw new RangeError('traceIds must be unique');
    if (!ids.length) return 0;
    await this.#ensure(false); this.#check();
    if (!this.#db) throw new RangeError('Unknown rejected trace ID');
    this.#transaction(db => {
      for (const id of ids) {
        const row = db.prepare('SELECT state FROM records WHERE trace_id=?').get(id);
        if (!row || row.state !== 'rejected') throw new RangeError('Only quarantined trace IDs may be recovered');
      }
      const sql = retry ? "UPDATE records SET state='pending',error=NULL WHERE trace_id=?" : 'DELETE FROM records WHERE trace_id=?';
      for (const id of ids) db.prepare(sql).run(id);
    });
    if (retry) this.#kick(true);
    return ids.length;
  }
  retryRejected(ids) { return this.#recover(ids, true); }
  discardRejected(ids) { return this.#recover(ids, false); }
  close({timeout = 5} = {}) {
    const duration = budget(timeout);
    if (this.#state === 'closed') return Promise.resolve();
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = this.#close(Date.now() + duration);
    return this.#closePromise;
  }
  async #close(deadline) {
    let timer;
    try {
      await Promise.race([
        (async () => { await Promise.allSettled([...this.#logs]); await this.#flush(deadline); })(),
        new Promise(resolve => { timer = setTimeout(resolve, Math.max(0, deadline - Date.now())); })
      ]);
    } finally {
      clearTimeout(timer);
      try { for (const target of this.#targets.keys()) this.#closedTargets.set(target, this.#status(target)); }
      catch (error) { this.#closeStorageError = error; }
      this.#state = 'closed'; clearTimeout(this.#timer);
      this.#controller?.abort(); this.#notify();
      if (this.#db) { this.#db.close(); this.#db = undefined; }
      process.off('beforeExit', this.#exit);
      if (this.#closeStorageError) throw this.#closeStorageError;
    }
  }
}
