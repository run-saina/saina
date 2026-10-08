export const DEFAULT_BASE_URL = 'https://api.saina.run';

/** Credit fields: decimal strings holding 64-bit integers, parsed to BigInt. */
export const CREDIT_FIELDS = Object.freeze(['balance', 'reserved', 'available', 'overdraft_allowance', 'debt',
  'credits', 'credits_charged', 'balance_after']);
const creditFields = new Set(CREDIT_FIELDS);
const UUID7 = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const INTEGER = /^-?[0-9]+$/;

/** A fresh RFC 9562 UUIDv7: 48-bit Unix ms timestamp, version 7, variant 10, random rest. */
export function newIdempotencyKey() {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  let ms = Date.now();
  for (let i = 5; i >= 0; i--) { bytes[i] = ms % 256; ms = Math.floor(ms / 256); }
  bytes[6] = 0x70 | (bytes[6] & 0x0f);
  bytes[8] = 0x80 | (bytes[8] & 0x3f);
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export class SainaHelmError extends Error {
  constructor(message, status, details = {}) {
    super(message);
    this.name = new.target.name;
    this.status = status;
    this.code = details.code ?? new.target.code ?? null;
    this.requestId = details.requestId ?? null;
    this.admitted = typeof details.admitted === 'boolean' ? details.admitted : null;
    this.state = details.state ?? null;
    this.retry = details.retry ?? null;
    this.retryAfter = details.retryAfter ?? null; // seconds
    this.suspension = details.suspension ?? null;
  }
}

/** The connection failed, timed out, or the response was cut off; the outcome may be unknown. */
export class SainaConnectionError extends SainaHelmError {}

const pascal = code => code.split('_').map(p => p[0].toUpperCase() + p.slice(1)).join('');
/** Error class per contract code; every class extends SainaHelmError. */
export const ERROR_CLASSES = Object.freeze(Object.fromEntries([
  'invalid_api_key', 'key_revoked', 'account_suspended', 'insufficient_credits', 'invalid_request',
  'rate_limited', 'overloaded', 'inference_failed', 'inference_unavailable', 'accounting_unavailable',
  'idempotency_conflict', 'request_in_progress', 'idempotency_result_expired', 'idempotency_unverifiable',
  'fresh_auth_required', 'permission_denied', 'not_found'
].map(code => {
  const cls = {[pascal(code)]: class extends SainaHelmError { static code = code; }}[pascal(code)];
  return [code, cls];
})));
export const {
  invalid_api_key: InvalidApiKey, key_revoked: KeyRevoked, account_suspended: AccountSuspended,
  insufficient_credits: InsufficientCredits, invalid_request: InvalidRequest, rate_limited: RateLimited,
  overloaded: Overloaded, inference_failed: InferenceFailed, inference_unavailable: InferenceUnavailable,
  accounting_unavailable: AccountingUnavailable, idempotency_conflict: IdempotencyConflict,
  request_in_progress: RequestInProgress, idempotency_result_expired: IdempotencyResultExpired,
  idempotency_unverifiable: IdempotencyUnverifiable, fresh_auth_required: FreshAuthRequired,
  permission_denied: PermissionDenied, not_found: NotFound
} = ERROR_CLASSES;

const RETRY_CODES = new Set(['accounting_unavailable', 'request_in_progress', 'rate_limited']);
const RETRY_IF_NOT_ADMITTED = new Set(['overloaded', 'inference_unavailable']);
const retryable = e => RETRY_CODES.has(e.code) || (RETRY_IF_NOT_ADMITTED.has(e.code) && e.admitted === false);

const toBigInt = value => typeof value === 'string' && INTEGER.test(value) ? BigInt(value) : null;

/** JSON.parse that turns known credit fields into BigInt without passing through a Number. */
export function parseCredits(text) {
  return JSON.parse(text, function (key, value, context) {
    if (!creditFields.has(key) || Array.isArray(this)) return value; // only object members, never array items
    if (typeof value === 'string' && INTEGER.test(value)) return BigInt(value);
    if (typeof value === 'number') {
      // Contract violation (credits must be strings): use the exact source text when the runtime provides it.
      if (context && typeof context.source === 'string' && INTEGER.test(context.source)) return BigInt(context.source);
      if (Number.isSafeInteger(value)) return BigInt(value);
      throw new SainaHelmError(`Unsafe numeric credit field ${key}`);
    }
    return value;
  });
}

function retryAfterSeconds(headers) {
  const value = headers?.get?.('Retry-After')?.trim();
  if (!value) return null;
  if (/^\d+$/.test(value)) return Number(value);
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, (date - Date.now()) / 1000);
}

/** Returns [error, typed]; old {"detail"} and non-JSON bodies still give a SainaHelmError with status. */
function httpError(status, text, headers) {
  let body = null;
  try { body = JSON.parse(text); } catch { /* not JSON */ }
  const retryAfter = retryAfterSeconds(headers);
  const requestId = headers?.get?.('X-Request-Id') ?? null;
  const error = body && typeof body === 'object' ? body.error : null;
  if (!error || typeof error !== 'object' || typeof error.code !== 'string') {
    const detail = body && typeof body.detail === 'string' && body.detail ? `: ${body.detail.slice(0, 500)}` : '';
    return [new SainaHelmError(`Saina HTTP ${status}${detail}`, status, {requestId, retryAfter}), false];
  }
  const message = typeof error.message === 'string' && error.message ? `: ${error.message.slice(0, 500)}` : '';
  const Cls = Object.hasOwn(ERROR_CLASSES, error.code) ? ERROR_CLASSES[error.code] : SainaHelmError;
  return [new Cls(`Saina HTTP ${status} ${error.code}${message}`, status, {
    code: error.code, requestId: error.request_id ?? requestId, admitted: error.admitted, state: error.state,
    retry: error.retry, retryAfter, suspension: error.suspension
  }), true];
}

class Uncertain { constructor(error, retryAfter = null) { this.error = error; this.retryAfter = retryAfter; } }

export class SainaHelm {
  /**
   * Inference calls are retried automatically only when they carry an Idempotency-Key: pass
   * {idempotencyKey} per call, or set autoIdempotencyKey so each logical call gets one new UUIDv7 key that
   * every retry of that call reuses with the identical body. Read-only account calls retry without a key.
   */
  constructor({baseUrl = DEFAULT_BASE_URL, apiKey, timeout = 60000, fetch: fetchImpl = globalThis.fetch,
               maxRetries = 2, maxRetryDelay = 30000, autoIdempotencyKey = false, sleep, random} = {}) {
    const url = new URL(baseUrl);
    if (!(url.protocol === 'https:' || (url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) || url.username || url.password || url.search || url.hash)
      throw new TypeError('Use HTTPS, or HTTP on localhost; no credentials/query in URL');
    if (typeof apiKey !== 'string' || !apiKey || /[\r\n]/.test(apiKey)) throw new TypeError('API key required');
    if (!Number.isFinite(timeout) || timeout <= 0) throw new TypeError('Positive timeout required');
    if (!Number.isInteger(maxRetries) || maxRetries < 0) throw new TypeError('maxRetries must be a nonnegative integer');
    if (!Number.isFinite(maxRetryDelay) || maxRetryDelay < 0) throw new TypeError('maxRetryDelay must be nonnegative');
    this.baseUrl = baseUrl.replace(/\/+$/, ''); this.apiKey = apiKey; this.timeout = timeout; this.fetch = fetchImpl;
    this.maxRetries = maxRetries; this.maxRetryDelay = maxRetryDelay; this.autoIdempotencyKey = autoIdempotencyKey;
    this.sleep = sleep ?? (ms => new Promise(resolve => setTimeout(resolve, ms)));
    this.random = random ?? Math.random;
    this.lastMetadata = null;
  }

  async ask(request, options) { return (await this.askWithMetadata(request, options)).result; }

  async askWithMetadata(request, {idempotencyKey} = {}) {
    if (!request || !request.questions || !Object.keys(request.questions).length)
      throw new TypeError('At least one typed question is required');
    const {result, metadata} = await this.#post('/v1/ask', request, idempotencyKey);
    validateAnswers(request, result);
    return {result, metadata};
  }

  async systemOne(request, options) { return (await this.systemOneWithMetadata(request, options)).result; }

  async systemOneWithMetadata(request, {idempotencyKey} = {}) {
    if (!request || !request.questions || !Object.keys(request.questions).length)
      throw new TypeError('At least one question is required');
    const {result, metadata} = await this.#post('/v1/systemone', request, idempotencyKey);
    const keys = Object.keys(request.questions);
    if (!result || typeof result.model !== 'string' || !result.answers || typeof result.answers !== 'object' ||
        Object.keys(result.answers).length !== keys.length || !keys.every(k => Object.hasOwn(result.answers, k)))
      throw new SainaHelmError('Invalid System One response');
    return {result, metadata};
  }

  /** GET /v1/account/balance; credit fields are BigInt. */
  balance() { return this.#get('/v1/account/balance', {}); }
  /** GET /v1/account/usage; filters: from, to, channel. */
  usage(filters = {}) { return this.#get('/v1/account/usage', filters); }
  /** GET /v1/account/requests; filters: cursor, limit, from, to, key_id, channel, status. */
  requests(filters = {}) { return this.#get('/v1/account/requests', filters); }

  async #post(path, request, idempotencyKey) {
    if (idempotencyKey == null && this.autoIdempotencyKey) idempotencyKey = newIdempotencyKey(); // once per call
    if (idempotencyKey != null && (typeof idempotencyKey !== 'string' || !UUID7.test(idempotencyKey.toLowerCase())))
      throw new TypeError('idempotencyKey must be a UUIDv7 string; use newIdempotencyKey()');
    const body = JSON.stringify(request); // serialized once: every retry sends identical bytes
    const headers = {'Content-Type': 'application/json'};
    if (idempotencyKey != null) headers['Idempotency-Key'] = idempotencyKey;
    const out = await this.#call('POST', path, body, headers, idempotencyKey ?? null, idempotencyKey != null, JSON.parse);
    this.lastMetadata = out.metadata;
    return out;
  }

  async #get(path, filters) {
    const query = new URLSearchParams();
    for (const [name, value] of Object.entries(filters))
      if (value != null) query.set(name, value instanceof Date ? value.toISOString() : String(value));
    const qs = query.toString();
    return (await this.#call('GET', path + (qs ? `?${qs}` : ''), undefined, {}, null, true, parseCredits)).result;
  }

  async #call(method, path, body, headers, idempotencyKey, retries, parse) {
    headers = {...headers, Accept: 'application/json', Authorization: `Bearer ${this.apiKey}`};
    let attempts = 0, waited = 0;
    for (;;) {
      attempts++;
      try {
        const {result, response} = await this.#attempt(method, path, body, headers, parse);
        const get = name => response.headers?.get?.(name) ?? null;
        const metadata = {
          requestId: get('X-Request-Id'), creditsCharged: toBigInt(get('X-Saina-Credits-Charged')),
          balance: toBigInt(get('X-Saina-Balance')), priceVersion: get('X-Saina-Price-Version'),
          replayed: (get('X-Saina-Replayed') ?? '').toLowerCase() === 'true', idempotencyKey, attempts
        };
        return {result, metadata};
      } catch (failure) {
        if (!(failure instanceof Uncertain)) throw failure;
        if (!retries || attempts > this.maxRetries) throw failure.error;
        const delay = failure.retryAfter != null ? failure.retryAfter * 1000
          : this.random() * Math.min(8000, 500 * 2 ** (attempts - 1));
        if (waited + delay > this.maxRetryDelay) throw failure.error;
        waited += delay;
        await this.sleep(delay);
      }
    }
  }

  async #attempt(method, path, body, headers, parse) {
    // Connection loss, timeout, or a body cut off: the outcome is unknown.
    const lost = () => { throw new Uncertain(new SainaConnectionError('Saina connection failed or timed out')); };
    let response, text;
    try {
      response = await this.fetch(`${this.baseUrl}${path}`, {
        method, redirect: 'error', signal: AbortSignal.timeout(this.timeout), headers, body
      });
    } catch { lost(); }
    if (!response.ok) {
      try { text = await response.text(); } catch { text = ''; }
      const [error, typed] = httpError(response.status, text, response.headers);
      if (retryable(error) || (response.status >= 500 && !typed)) throw new Uncertain(error, error.retryAfter);
      throw error;
    }
    try { text = await response.text(); } catch { lost(); }
    try { return {result: parse(text), response}; }
    catch (e) {
      if (e instanceof SainaHelmError) throw e;
      throw new Uncertain(new SainaHelmError('Invalid JSON response'));
    }
  }
}

export { SainaHelm as Saina, SainaHelmError as SainaError };

function validateAnswers(request, result) {
  const probability = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
  const fail = () => { throw new SainaHelmError('Invalid typed answer response'); };
  if (!result || typeof result.model !== 'string' || !result.answers ||
      Object.keys(result.answers).length !== Object.keys(request.questions).length) fail();
  for (const [key, q] of Object.entries(request.questions)) {
    const a = Object.hasOwn(result.answers, key) && result.answers[key];
    if (!a || a.type !== q.type) fail();
    const labels = q.type === 'yes_no' ? ['yes', 'no'] : q.type === 'rating' ? q.levels.map((_, i) => String(i)) : Object.keys(q.options);
    const vector = q.type === 'yes_no' ? {yes: a.yes, no: a.no} : q.type === 'multi_choice' ? a.memberships : a.probabilities;
    if (!vector || Object.keys(vector).length !== labels.length || !labels.every(k => Object.hasOwn(vector, k) && probability(vector[k]))) fail();
    if (q.type !== 'multi_choice' && (Math.abs(labels.reduce((s, k) => s + vector[k], 0) - 1) > 1e-5 || !probability(a.confidence))) fail();
    if (q.type === 'single_choice' && !(labels.includes(a.selection) || request.mode === 'decision' && a.selection === null)) fail();
    if (q.type === 'rating' && (typeof a.expected_level !== 'number' || !Number.isFinite(a.expected_level) || a.expected_level < 0 || a.expected_level > labels.length - 1 || !Array.isArray(a.levels) || a.levels.length !== labels.length)) fail();
    if (request.mode === 'decision') {
      if (!['accepted', 'below_threshold', 'below_margin', 'tie'].includes(a.reason)) fail();
      if (q.type === 'yes_no' && !(a.selected === null || typeof a.selected === 'boolean')) fail();
      if (q.type === 'rating' && !(a.level === null || Number.isInteger(a.level) && a.level >= 0 && a.level < labels.length)) fail();
      if (q.type === 'multi_choice' && (!Array.isArray(a.selections) || new Set(a.selections).size !== a.selections.length || !a.selections.every(k => labels.includes(k)))) fail();
    }
  }
}
