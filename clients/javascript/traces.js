/** Explicit decision evidence capture; no model is called unless invokeEndpoint is used. */
export class TraceDeliveryError extends Error {
  constructor(message, {status, code} = {}) {
    super(message); this.name = 'TraceDeliveryError'; this.status = status; this.code = code;
  }
}
const identifier = value => {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_.-]{1,120}$/.test(value)) throw new TypeError('Use a stable identifier of 1–120 letters, digits, dots, underscores or hyphens');
  return value;
};
const timestampPattern = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/;
const timestamp = value => {
  const match = typeof value === 'string' ? timestampPattern.exec(value) : null;
  if (!match) throw new TypeError('recorded_at must be an ISO 8601 timestamp with a timezone');
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  const hour = Number(match[4]), minute = Number(match[5]), second = Number(match[6]);
  const zone = match[8];
  const probe = new Date(Date.UTC(year, month - 1, day));
  const offsetHours = zone === 'Z' ? 0 : Number(zone.slice(1, 3));
  const offsetMinutes = zone === 'Z' ? 0 : Number(zone.slice(4, 6));
  if (hour > 23 || minute > 59 || second > 59 || offsetHours > 23 || offsetMinutes > 59 ||
      probe.getUTCFullYear() !== year || probe.getUTCMonth() !== month - 1 || probe.getUTCDate() !== day) {
    throw new TypeError('recorded_at must be an ISO 8601 timestamp with a timezone');
  }
  return value;
};
const snapshot = value => JSON.parse(JSON.stringify(value, (_key, v) => {
  if (v === undefined || typeof v === 'function' || typeof v === 'symbol' || typeof v === 'bigint' || (typeof v === 'number' && !Number.isFinite(v))) throw new TypeError('Trace values must be finite JSON data');
  return v;
}));
const bytes = value => new TextEncoder().encode(JSON.stringify(value)).length;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let reportOutcomeWarned = false;

export class DecisionTrace {
  #record; #finished = false; #hasAnswer = false;
  constructor({task, inputs, decisionId, metadata = {}, recorded_at} = {}) {
    if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)) throw new TypeError('Trace metadata must be an object');
    // Undefined keeps the current clock. A supplied value replays a simulated clock.
    const recorded = recorded_at === undefined ? new Date().toISOString() : timestamp(recorded_at);
    this.decisionId = identifier(decisionId ?? 'trace_' + crypto.randomUUID().replaceAll('-', ''));
    this.#record = {schema_version: '1.0', decision_id: this.decisionId, task: identifier(task),
      inputs: snapshot(inputs), metadata: snapshot(metadata), recorded_at: recorded};
  }
  setAnswer(answer) {
    if (this.#finished) throw new Error('Decision scope is already finished');
    this.#record.answer = snapshot(answer); this.#hasAnswer = true;
  }
  finish(start, failed, error) {
    this.#finished = true;
    Object.assign(this.#record, {completed_at: new Date().toISOString(), timing: {total_ms: performance.now() - start},
      status: failed ? 'error' : this.#hasAnswer ? 'completed' : 'incomplete'});
    if (failed) this.#record.error = {type: error instanceof Error ? error.name.slice(0, 200) || 'Error' : 'ThrownValue'};
    return snapshot(this.#record);
  }
}

export class TraceClient {
  #apiKey; #queue = []; #sizes = []; #timer; #activeFlush; #retryAfter = 0; #closed = false; #closing = false;
  constructor({collection, apiKey, baseUrl = 'https://api.saina.run', source = 'application', batchSize = 100,
    flushInterval = 5000, maxPending = 1000, timeoutMs = 10000, maxRetries = 2, onError} = {}) {
    const url = new URL(baseUrl);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new TypeError('Use an explicit HTTP(S) service origin');
    if (apiKey !== undefined && (typeof apiKey !== 'string' || !apiKey || /[\r\n]/.test(apiKey))) throw new TypeError('Invalid API key');
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 100) throw new TypeError('batchSize must be 1–100');
    if (!Number.isInteger(maxPending) || maxPending < batchSize) throw new TypeError('maxPending must be at least batchSize');
    if (![flushInterval, timeoutMs].every(v => Number.isFinite(v) && v > 0)) throw new TypeError('flushInterval and timeoutMs must be positive finite numbers');
    if (!Number.isInteger(maxRetries) || maxRetries < 0 || maxRetries > 5) throw new TypeError('maxRetries must be 0–5');
    if (onError !== undefined && typeof onError !== 'function') throw new TypeError('onError must be a function');
    this.collection = identifier(collection); this.source = identifier(source); this.baseUrl = url.origin; this.#apiKey = apiKey;
    Object.assign(this, {batchSize, flushInterval, maxPending, timeoutMs, maxRetries, onError});
    this.lastError = null; this.droppedCount = 0;
  }
  get pendingCount() { return this.#queue.length; }
  #checkOpen() { if (this.#closed || this.#closing) throw new Error('Trace client is closed'); }
  #notify(error) {
    this.lastError = error;
    try { if (this.onError) this.onError(error); else console.warn('Saina trace capture/delivery failed; inspect lastError and pendingCount'); } catch { /* Preserve application behavior. */ }
  }
  #schedule() {
    if (this.#closing || !this.#queue.length) return;
    if (this.#timer) {
      if (this.#queue.length < this.batchSize) return;
      clearTimeout(this.#timer);
    }
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      void this.flush().catch(error => {
        this.#retryAfter = Date.now() + this.flushInterval;
        this.#notify(error);
      }).finally(() => this.#schedule());
    }, Math.max(this.#queue.length >= this.batchSize ? 0 : this.flushInterval, this.#retryAfter - Date.now()));
    this.#timer.unref?.();
  }
  #enqueue(record) {
    this.#checkOpen();
    const size=bytes(record);
    if (size > 1048576 || this.#queue.length >= this.maxPending) {
      this.droppedCount++; throw new TraceDeliveryError('Trace exceeds 1 MiB or the pending queue is full');
    }
    this.#queue.push(record); this.#sizes.push(size); this.#schedule();
  }
  async decision(options, fn) {
    this.#checkOpen();
    const trace = new DecisionTrace(options), start = performance.now();
    let failed = false, error;
    try { return await fn(trace); }
    catch (caught) { failed = true; error = caught; throw caught; }
    finally {
      try { this.#enqueue(trace.finish(start, failed, error)); }
      catch (deliveryError) { this.#notify(deliveryError); }
    }
  }
  async #post(path, body, retry = true, retryNotFound = false) {
    const payload = JSON.stringify(body), attempts = retry ? this.maxRetries : 0;
    for (let attempt = 0; ; attempt++) {
      let delay = Math.min(250 * 2 ** attempt, 5000), transient = true, error;
      try {
        const response = await fetch(this.baseUrl + path, {method: 'POST', redirect: 'error',
          headers: {'Content-Type': 'application/json', 'X-Saina-Client': 'javascript-sdk', ...(this.#apiKey ? {Authorization: 'Bearer ' + this.#apiKey} : {})},
          body: payload, signal: AbortSignal.timeout(this.timeoutMs)});
        const reader = response.body?.getReader();
        let length = 0; const chunks = [];
        if (!reader) throw new TraceDeliveryError('Trace service returned an empty response');
        try {
          while (true) {
            const {done, value} = await reader.read(); if (done) break;
            length += value.length;
            if (length > 2 * 1048576) { await reader.cancel(); throw new TraceDeliveryError('Trace service response exceeds 2 MiB'); }
            chunks.push(value);
          }
        } finally { reader.releaseLock(); }
        const data = new Uint8Array(length); let offset = 0;
        for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }
        let value; try { value = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(data)); } catch { value = null; }
        if (response.ok && value && typeof value === 'object') return value;
        error = new TraceDeliveryError('Saina trace request failed', {status: response.status, code: value?.error?.code});
        transient = response.ok || [408, 429, 500, 502, 503, 504].includes(response.status) || retryNotFound && response.status === 404;
        const after = response.headers.get('Retry-After');
        if (after !== null && Number.isFinite(Number(after))) delay = Math.min(30000, Math.max(0, Number(after) * 1000));
      } catch (caught) { error = caught instanceof TraceDeliveryError ? caught : new TraceDeliveryError('Saina trace connection or response failed'); }
      if (!retry || !transient || attempt >= attempts) throw error;
      await sleep(delay);
    }
  }
  async flush() {
    // Serialize overlapping flushes, including calls made during an in-flight batch.
    if (this.#activeFlush) { await this.#activeFlush; return this.flush(); }
    const count = this.#queue.length;
    this.#activeFlush = this.#deliver(count);
    try { await this.#activeFlush; } catch (error) { this.lastError = error; throw error; } finally { this.#activeFlush = undefined; }
  }
  async #deliver(remaining) {
    while (remaining > 0) {
      const batch = []; let size = 0;
      for (let i=0; i<Math.min(this.#queue.length,this.batchSize,remaining); i++) {
        const record=this.#queue[i], n=this.#sizes[i]; if (batch.length && size + n > 8 * 1048576) break;
        batch.push(record); size += n;
      }
      const receipt = await this.#post('/v1/studio/data/collections/' + encodeURIComponent(this.collection) + '/traces', {source: this.source, traces: batch});
      if (JSON.stringify(receipt.decision_ids) !== JSON.stringify(batch.map(r => r.decision_id))) throw new TraceDeliveryError('Trace service returned an invalid acknowledgement');
      this.#queue.splice(0, batch.length); this.#sizes.splice(0,batch.length); remaining -= batch.length;
    }
    this.lastError = null; this.#retryAfter = 0;
  }
  async close() {
    this.#closing = true; clearTimeout(this.#timer); this.#timer = undefined;
    await this.flush(); this.#closed = true;
  }
  /**
   * Record a business outcome: the business value of a decision, or a proxy for it
   * (resolved, revenue, cost, CSAT). A business outcome is not a label.
   * Posts to `/v1/studio/business-outcomes`. `/v1/studio/outcomes` remains a deprecated server alias.
   */
  async reportBusinessOutcome(decisionId, {values, eventId}) {
    this.#checkOpen();
    await this.flush();
    return this.#post('/v1/studio/business-outcomes', {
      source: this.source, collection_id: this.collection, decision_id: identifier(decisionId),
      event_id: identifier(eventId), values: snapshot(values),
    }, true, true);
  }
  /** @deprecated Use {@link TraceClient#reportBusinessOutcome}. */
  async reportOutcome(decisionId, businessOutcome) {
    if (!reportOutcomeWarned) {
      reportOutcomeWarned = true;
      console.warn('TraceClient.reportOutcome() is deprecated; use reportBusinessOutcome().');
    }
    return this.reportBusinessOutcome(decisionId, businessOutcome);
  }
  async invokeEndpoint(endpoint, request, {version = 'production'} = {}) {
    this.#checkOpen();
    if (version !== 'production' && (!Number.isInteger(version) || version < 1)) throw new TypeError('Choose production or a positive version');
    if (!request || typeof request !== 'object' || Array.isArray(request) || 'version' in request || 'trace_collection_id' in request) throw new TypeError('Pass an endpoint request without version or trace_collection_id');
    return this.#post('/v1/studio/endpoints/' + encodeURIComponent(identifier(endpoint)) + '/decide',
      {...snapshot(request), version, trace_collection_id: this.collection}, false);
  }
}
