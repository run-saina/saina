export {Traces} from './durable-traces.js';
export {TraceClient, DecisionTrace, TraceDeliveryError} from './traces.js';
import {Answers, validateTyped} from './answers.js';
import {nativeRequest, normalizeRequest, flattenExtra, requestFields, questionFields, jsonValue} from './validation.js';
export {Answers, Selections, YesNoAnswer, SingleChoiceAnswer, RatingAnswer, MultiChoiceAnswer, YesNoSelection, SingleChoiceSelection, RatingSelection, MultiChoiceSelection} from './answers.js';

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

export class SainaError extends Error {
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
    this.raw = details.raw ?? null;
    this.fields = details.fields ?? {};
  }
}

export class HTTPError extends SainaError {}
export class AuthenticationError extends HTTPError {}
export class PermissionDeniedError extends HTTPError {}
export class NotFoundError extends HTTPError {}
export class InvalidRequestError extends HTTPError {}
export class RateLimitError extends HTTPError {}
export class IdempotencyError extends HTTPError {}
export class ServerError extends HTTPError {}
export class ModelBusyError extends ServerError {}
export class TransportError extends SainaError {
  constructor(message, status, details = {}) { super(message, status, details); this.timedOut = details.timedOut ?? false; }
}
export class ProtocolError extends SainaError {}
export class SpoolFullError extends SainaError {}
export class TraceStorageError extends SainaError {}
export const SainaHelmError = SainaError;
export const SainaConnectionError = TransportError;
const codeParents = {
  invalid_api_key: AuthenticationError,
  key_revoked: AuthenticationError,
  fresh_auth_required: AuthenticationError,
  permission_denied: PermissionDeniedError,
  account_suspended: PermissionDeniedError,
  not_found: NotFoundError,
  invalid_request: InvalidRequestError,
  rate_limited: RateLimitError,
  insufficient_credits: HTTPError,
  idempotency_conflict: IdempotencyError,
  request_in_progress: IdempotencyError,
  idempotency_result_expired: IdempotencyError,
  idempotency_unverifiable: IdempotencyError,
  inference_failed: ServerError,
  accounting_unavailable: ServerError,
  overloaded: ModelBusyError,
  inference_unavailable: ModelBusyError,
};
const statusClass = status => ({401: AuthenticationError, 403: PermissionDeniedError, 404: NotFoundError,
  422: InvalidRequestError, 429: RateLimitError, 503: ModelBusyError}[status] ??
  (status >= 500 && status < 600 ? ServerError : HTTPError));
const pascal = code => code.split('_').map(p => p[0].toUpperCase() + p.slice(1)).join('');
/** Error class per contract code; classes inherit their broad HTTP category. */
export const ERROR_CLASSES = Object.freeze(Object.fromEntries([
  'invalid_api_key', 'key_revoked', 'account_suspended', 'insufficient_credits', 'invalid_request',
  'rate_limited', 'overloaded', 'inference_failed', 'inference_unavailable', 'accounting_unavailable',
  'idempotency_conflict', 'request_in_progress', 'idempotency_result_expired', 'idempotency_unverifiable',
  'fresh_auth_required', 'permission_denied', 'not_found'
].map(code => {
  const cls = {[pascal(code)]: class extends codeParents[code] { static code = code; }}[pascal(code)];
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
    return [new (statusClass(status))(`Saina HTTP ${status}${detail}`, status, {requestId, retryAfter, raw: body, fields: error}), false];
  }
  const message = typeof error.message === 'string' && error.message ? `: ${error.message.slice(0, 500)}` : '';
  const Cls = Object.hasOwn(ERROR_CLASSES, error.code) ? ERROR_CLASSES[error.code] : statusClass(status);
  return [new Cls(`Saina HTTP ${status} ${error.code}${message}`, status, {
    code: error.code, requestId: error.request_id ?? requestId, admitted: error.admitted, state: error.state,
    retry: error.retry, retryAfter, suspension: error.suspension, raw: body, fields: error
  }), true];
}

// -- images and audio ------------------------------------------------------------------------------
/** Per request: at most 8 images and 4 audio clips, each at most 10,000,000 bytes (the server checks too). */
export const MEDIA_LIMITS = Object.freeze({images: 8, audios: 4, bytes: 10_000_000, audioSeconds: 10});
const IMAGE_TYPES = ['png', 'jpeg', 'webp'];
const AUDIO_TYPES = ['wav', 'x-wav', 'flac', 'mpeg', 'ogg', 'webm'];

function sniffImage(b) {
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'png';
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'jpeg';
  if (ascii(b, 0, 4) === 'RIFF' && ascii(b, 8, 12) === 'WEBP') return 'webp';
  return null;
}
function sniffAudio(b) {
  if (ascii(b, 0, 4) === 'RIFF' && ascii(b, 8, 12) === 'WAVE') return 'wav';
  if (ascii(b, 0, 4) === 'fLaC') return 'flac';
  if (ascii(b, 0, 4) === 'OggS') return 'ogg';
  if (b[0] === 0x1a && b[1] === 0x45 && b[2] === 0xdf && b[3] === 0xa3) return 'webm';
  if (ascii(b, 0, 3) === 'ID3' || (b[0] === 0xff && (b[1] & 0xe0) === 0xe0)) return 'mpeg';
  return null;
}
const ascii = (b, start, end) => String.fromCharCode(...b.subarray(start, end));

function base64(bytes) {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  let text = '';
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

async function mediaBytes(value, what, fetchImpl) {
  if (typeof value === 'string' && /^https?:\/\//i.test(value)) {
    const response = await fetchImpl(value);
    if (!response.ok) throw new TypeError(`Cannot download the ${what} at ${value} (HTTP ${response.status})`);
    return {bytes: new Uint8Array(await response.arrayBuffer()), type: response.headers?.get?.('Content-Type') ?? ''};
  }
  if (typeof Blob !== 'undefined' && value instanceof Blob) return {bytes: new Uint8Array(await value.arrayBuffer()), type: value.type};
  if (value instanceof ArrayBuffer) return {bytes: new Uint8Array(value), type: ''};
  if (ArrayBuffer.isView(value)) return {bytes: new Uint8Array(value.buffer, value.byteOffset, value.byteLength), type: ''};
  throw new TypeError(`An ${what} is a data URL, an http(s) URL, a Blob or File, an ArrayBuffer or a typed array (Buffer)`);
}

async function toDataUrl(value, kind, fetchImpl = globalThis.fetch) {
  if (typeof value === 'string' && value.slice(0, 5).toLowerCase() === 'data:') return value;
  const what = kind === 'image' ? 'image' : 'audio clip';
  const {bytes} = await mediaBytes(value, what, fetchImpl);
  if (bytes.length > MEDIA_LIMITS.bytes) throw new TypeError(`The ${what} is larger than ${MEDIA_LIMITS.bytes} bytes`);
  const subtype = kind === 'image' ? sniffImage(bytes) : sniffAudio(bytes);
  if (!subtype) throw new TypeError(kind === 'image' ? 'Send PNG, JPEG or WebP images' : 'Send WAV, FLAC, MP3, Ogg or WebM audio');
  return `data:${kind}/${subtype};base64,${base64(bytes)}`;
}

/** One image as a data:image/(png|jpeg|webp);base64 URL, from a data URL, an http(s) URL, a Blob/File or bytes. */
export function imageDataUrl(value, {fetch: fetchImpl} = {}) { return toDataUrl(value, 'image', fetchImpl); }
/** One audio clip (WAV, FLAC, MP3, Ogg or WebM) as a data:audio/...;base64 URL, from the same kinds of value. */
export function audioDataUrl(value, {fetch: fetchImpl} = {}) { return toDataUrl(value, 'audio', fetchImpl); }

/** The request with images and audios converted to data URLs (other fields untouched). */
async function withMedia(request, fetchImpl) {
  const out = {...request};
  for (const [name, kind] of [['images', 'image'], ['audios', 'audio']]) {
    if (request[name] == null) continue;
    if (!Array.isArray(request[name])) throw new TypeError(`${name} must be an array`);
    out[name] = await Promise.all(request[name].map(v => toDataUrl(v, kind, fetchImpl)));
  }
  return out;
}

function validateIdempotencyKey(key) {
  if (key != null && (typeof key !== 'string' || key.length !== 36 || !UUID7.test(key.toLowerCase())))
    throw new TypeError('idempotencyKey must be a UUIDv7 string; use newIdempotencyKey()');
}

class Uncertain { constructor(error, retryAfter = null) { this.error = error; this.retryAfter = retryAfter; } }

export class SainaHelm {
  /**
   * Inference calls are retried automatically only when they carry an Idempotency-Key: pass
   * {idempotencyKey} per call, or set autoIdempotencyKey so each logical call gets one new UUIDv7 key that
   * every retry of that call reuses with the identical body. Read-only account calls retry without a key.
   */
  constructor({baseUrl, apiKey, timeout, allowHttp = false, fetch: fetchImpl = globalThis.fetch,
               maxRetries = 2, maxRetryDelay = 30000, autoIdempotencyKey = true, sleep, random} = {}) {
    const env = globalThis.process?.env ?? {};
    baseUrl ??= env.SAINA_BASE_URL ?? DEFAULT_BASE_URL;
    apiKey ??= env.SAINA_API_KEY;
    timeout ??= env.SAINA_TIMEOUT === undefined ? 60000 : Number(env.SAINA_TIMEOUT) * 1000;
    if (typeof baseUrl !== 'string' || /\s/.test(baseUrl)) throw new TypeError('Invalid service URL');
    if (typeof allowHttp !== 'boolean' || typeof autoIdempotencyKey !== 'boolean') throw new TypeError('Boolean options required');
    const url = new URL(baseUrl);
    if (!(url.protocol === 'https:' || (url.protocol === 'http:' && (allowHttp || ['localhost', '[::1]'].includes(url.hostname) || /^127(?:\.\d{1,3}){3}$/.test(url.hostname)))) || url.username || url.password || url.search || url.hash)
      throw new TypeError('Use HTTPS, or HTTP on localhost; no credentials/query in URL');
    if (typeof apiKey !== 'string' || !apiKey.trim() || /[\r\n]/.test(apiKey)) throw new TypeError('API key required');
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
    validateIdempotencyKey(idempotencyKey);
    request = await withMedia(nativeRequest(request), this.fetch);
    const {result, metadata} = await this.#post('/v1/ask', request, idempotencyKey);
    validateTyped(request, result);
    const bodyId = result.request_id;
    if (bodyId != null && (typeof bodyId !== 'string' || !bodyId.trim())) throw new ProtocolError('Invalid request ID');
    if (bodyId != null && metadata.requestId != null && bodyId !== metadata.requestId)
      throw new ProtocolError('Body and header request IDs disagree');
    metadata.requestId = bodyId ?? metadata.requestId;
    if (typeof metadata.requestId !== 'string' || !metadata.requestId.trim()) throw new ProtocolError('Typed answers require a request ID');
    const typed = new Answers(result, metadata, request.questions);
    this.lastMetadata = typed.metadata;
    return {result: typed, metadata: typed.metadata};
  }

  async decisions(request, options) { return (await this.decisionsWithMetadata(request, options)).result; }

  async decisionsWithMetadata(request, {idempotencyKey} = {}) {
    validateIdempotencyKey(idempotencyKey);
    if (!Array.isArray(request?.questions) || !request.questions.length)
      throw new TypeError('At least one question is required');
    request = flattenExtra(request, requestFields);
    request.questions = request.questions.map(q => flattenExtra(q, questionFields));
    const {result, metadata} = await this.#post('/v1/decisions', request, idempotencyKey);
    const probability = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
    if (typeof result?.model !== 'string' || !Array.isArray(result.answers) ||
        result.answers.length !== request.questions.length || result.answers.some((a, i) =>
          !a || a.name !== (request.questions[i].name ?? null) || ![request.questions[i].type, 'refusal'].includes(a.type) ||
          a.type === 'predicate' && !probability(a.probability) ||
          ['choice', 'score'].includes(a.type) && (!Array.isArray(a.probabilities) || !probability(a.confidence))))
      throw new ProtocolError('Invalid Decisions response');
    return {result, metadata};
  }

  async systemOne(request, options) { return (await this.systemOneWithMetadata(request, options)).result; }

  async systemOneWithMetadata(request, {idempotencyKey} = {}) {
    validateIdempotencyKey(idempotencyKey);
    if (!request || !request.questions || !Object.keys(request.questions).length)
      throw new TypeError('At least one question is required');
    request = await withMedia(normalizeRequest(request), this.fetch);
    const {result, metadata} = await this.#post('/v1/systemone', request, idempotencyKey);
    const keys = Object.keys(request.questions);
    if (!result || typeof result.model !== 'string' || !result.answers || typeof result.answers !== 'object' ||
        Object.keys(result.answers).length !== keys.length || !keys.every(k => Object.hasOwn(result.answers, k)))
      throw new ProtocolError('Invalid System One response');
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
    validateIdempotencyKey(idempotencyKey);
    jsonValue(request);
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
    const lost = error => { throw new Uncertain(new TransportError('Saina connection failed or timed out', undefined, {timedOut: error?.name === 'TimeoutError'})); };
    let response, text;
    try {
      response = await this.fetch(`${this.baseUrl}${path}`, {
        method, redirect: 'error', signal: AbortSignal.timeout(this.timeout), headers, body
      });
    } catch (error) { lost(error); }
    if (!response.ok) {
      try { text = await response.text(); } catch { text = ''; }
      const [error, typed] = httpError(response.status, text, response.headers);
      if (retryable(error) || (response.status >= 500 && !typed)) throw new Uncertain(error, error.retryAfter);
      throw error;
    }
    try { text = await response.text(); } catch (error) { lost(error); }
    try { return {result: parse(text), response}; }
    catch (e) {
      if (e instanceof SainaHelmError) throw e;
      throw new ProtocolError('Invalid JSON response');
    }
  }
}

export { SainaHelm as Saina };

/** JSON serialization that preserves arbitrary credit amounts as decimal strings. */
export function stringifyCredits(value) {
  return JSON.stringify(value, (_key, item) => typeof item === 'bigint' ? item.toString() : item);
}
