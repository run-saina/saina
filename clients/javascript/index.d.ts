export type Content = string | {[key: string]: Json} | Json[];
export type Json = null | boolean | number | string | Json[] | {[key: string]: Json};
export type Options = Record<string, Content | null> | string[];
export type Question = {extra?: Record<string, Json>} & (
  | {type: 'yes_no'; question: Content; descriptions?: {yes: Content; no: Content}}
  | {type: 'single_choice'; question: Content; options: Options}
  | {type: 'rating'; question: Content; levels: Content[]}
  | {type: 'multi_choice'; question: Content; options: Options});
export type DecisionQuestion = Question extends infer Q ? Q extends Question ? Q & {threshold?: number} &
  (Q['type'] extends 'yes_no' | 'single_choice' ? {min_margin?: number} : {}) : never : never;
/**
 * An image: a `data:image/(png|jpeg|webp);base64,...` URL, an http(s) URL (downloaded by the client), a Blob or
 * File, an ArrayBuffer or a typed array such as a Node Buffer. Sent as a data URL.
 */
export type ImageInput = string | Blob | ArrayBuffer | ArrayBufferView;
/** An audio clip (WAV, FLAC, MP3, Ogg or WebM, at most 10 s): the same kinds of value as {@link ImageInput}. */
export type AudioInput = string | Blob | ArrayBuffer | ArrayBufferView;
/**
 * Images and audio clips for a model that reads them (see `inputs` in /v1/models); a text-only model refuses them
 * with `invalid_request`. At most 8 images and 4 clips, each at most 10,000,000 bytes. Images go before the text,
 * read at up to 1.6 megapixels; each clip is read as a log-mel spectrogram after the images. Every question sees
 * all of them, and their input tokens count in `usage.input_tokens`.
 */
export interface MediaFields {images?: ImageInput[]; audios?: AudioInput[]; extra?: Record<string, Json>}
/** `saina-helm`: whichever Helm the endpoint serves. `helm-0.8b` is the deprecated 0.1 name of `saina-helm-0.8b`. */
export type AskRequest = {model: string; state: Content; questions: Record<string, Question>; specialist?: string} & MediaFields;
export type SelectionReason = 'accepted' | 'below_threshold' | 'tie' | 'below_margin' | 'no_options_selected';
export interface SelectOptions {threshold: number; minMargin?: number}
export interface Serializable {toObject(): Record<string, Json>; toJSON(): Record<string, Json>}
export class YesNoAnswer implements Serializable {
  readonly type: 'yes_no'; readonly yes: number; readonly no: number; readonly confidence: number;
  select(options: SelectOptions): YesNoSelection;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class SingleChoiceAnswer implements Serializable {
  readonly type: 'single_choice'; readonly probabilities: Readonly<Record<string, number>>; readonly confidence: number;
  select(options: SelectOptions): SingleChoiceSelection;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class RatingAnswer implements Serializable {
  readonly type: 'rating'; readonly probabilities: Readonly<Record<number, number>>;
  readonly levels: readonly Content[]; readonly expected_level: number; readonly confidence: number;
  select(options: SelectOptions): RatingSelection;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class MultiChoiceAnswer implements Serializable {
  readonly type: 'multi_choice'; readonly memberships: Readonly<Record<string, number>>;
  select(options: SelectOptions): MultiChoiceSelection;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export type Answer = YesNoAnswer | SingleChoiceAnswer | RatingAnswer | MultiChoiceAnswer;
export interface Selection extends Serializable {
  readonly type: Answer['type']; readonly value: boolean | string | number | readonly string[] | null;
  readonly accepted: boolean; readonly reason: SelectionReason; readonly answer: Answer;
  readonly threshold: number; readonly minMargin: number;
}
export class YesNoSelection implements Selection {
  readonly type: 'yes_no'; readonly value: boolean | null; readonly answer: YesNoAnswer;
  readonly accepted: boolean; readonly reason: SelectionReason; readonly threshold: number; readonly minMargin: number;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class SingleChoiceSelection implements Selection {
  readonly type: 'single_choice'; readonly value: string | null; readonly answer: SingleChoiceAnswer;
  readonly accepted: boolean; readonly reason: SelectionReason; readonly threshold: number; readonly minMargin: number;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class RatingSelection implements Selection {
  readonly type: 'rating'; readonly value: number | null; readonly label: Content | null; readonly answer: RatingAnswer;
  readonly accepted: boolean; readonly reason: SelectionReason; readonly threshold: number; readonly minMargin: number;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export class MultiChoiceSelection implements Selection {
  readonly type: 'multi_choice'; readonly value: readonly string[]; readonly answer: MultiChoiceAnswer;
  readonly accepted: boolean; readonly reason: SelectionReason; readonly threshold: number; readonly minMargin: number;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export type Selections = Readonly<Record<string, Selection>> & Serializable;
export const Selections: {new(items: Record<string, Selection>): Selections};
export class Answers implements Serializable {
  readonly requestId: string; readonly model: string; readonly specialist: string | null;
  readonly usage: Readonly<{input_tokens?: number; output_tokens?: number}>;
  readonly metadata: Readonly<ResponseMetadata>; readonly raw: unknown;
  readonly answers: Readonly<Record<string, Answer>>;
  select(options: SelectOptions): Selections;
  toObject(): Record<string, Json>; toJSON(): Record<string, Json>;
}
export type AskResponse = Answers;
export type SystemOneQuestion = {type: 'choice' | 'score' | 'noul'; instructions: Content; extra?: Record<string, Json>; criteria?: {[key: string]: Json} | Json[] | null};
export type SystemOneRequest = {model: string; state: Content; questions: Record<string, SystemOneQuestion>} & MediaFields;
export type SystemOneAnswer =
  | {type: 'noul'; noul: number}
  | {type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number}
  | {type: 'score'; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number};
export interface SystemOneResponse {model: string; answers: Record<string, SystemOneAnswer>; usage: {input_tokens: number; output_tokens: number}}

export const DEFAULT_BASE_URL: 'https://api.saina.run';
export const CREDIT_FIELDS: readonly string[];
/** A fresh RFC 9562 UUIDv7 string. */
export function newIdempotencyKey(): string;
/** JSON.parse that converts known credit fields (decimal strings) to BigInt. */
export function parseCredits(text: string): any;
export const MEDIA_LIMITS: Readonly<{images: 8; audios: 4; bytes: 10000000; audioSeconds: 10}>;
/** One image as a base64 data URL (type from its content). */
export function imageDataUrl(value: ImageInput, options?: {fetch?: typeof fetch}): Promise<string>;
/** One audio clip as a base64 data URL (type from its content). */
export function audioDataUrl(value: AudioInput, options?: {fetch?: typeof fetch}): Promise<string>;

export type ErrorCode =
  | 'invalid_api_key' | 'key_revoked' | 'account_suspended' | 'insufficient_credits' | 'invalid_request'
  | 'rate_limited' | 'overloaded' | 'inference_failed' | 'inference_unavailable' | 'accounting_unavailable'
  | 'idempotency_conflict' | 'request_in_progress' | 'idempotency_result_expired' | 'idempotency_unverifiable'
  | 'fresh_auth_required' | 'permission_denied' | 'not_found';
export interface Suspension {types: Array<'financial' | 'security'>; next_step: 'buy_credits' | 'contact_support'}
export class SainaError extends Error {
  constructor(message: string, status?: number, details?: Record<string, unknown>);
  readonly status?: number;
  readonly code: ErrorCode | string | null;
  readonly requestId: string | null;
  readonly admitted: boolean | null;
  readonly state: 'not_admitted' | 'in_progress' | 'terminal' | 'unknown' | null;
  readonly retry: 'same_operation' | 'new_operation' | 'no' | null;
  /** Seconds from Retry-After. */
  readonly retryAfter: number | null;
  readonly suspension: Suspension | null;
  readonly raw: unknown;
  readonly fields: Record<string, unknown>;
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
export class TransportError extends SainaError {readonly timedOut: boolean}
export class ProtocolError extends SainaError {}
export class SpoolFullError extends SainaError {}
export class TraceStorageError extends SainaError {}
export class InvalidApiKey extends AuthenticationError {}
export class KeyRevoked extends AuthenticationError {}
export class FreshAuthRequired extends AuthenticationError {}
export class PermissionDenied extends PermissionDeniedError {}
export class AccountSuspended extends PermissionDeniedError {}
export class NotFound extends NotFoundError {}
export class InvalidRequest extends InvalidRequestError {}
export class RateLimited extends RateLimitError {}
export class InsufficientCredits extends HTTPError {}
export class IdempotencyConflict extends IdempotencyError {}
export class RequestInProgress extends IdempotencyError {}
export class IdempotencyResultExpired extends IdempotencyError {}
export class IdempotencyUnverifiable extends IdempotencyError {}
export class InferenceFailed extends ServerError {}
export class AccountingUnavailable extends ServerError {}
export class Overloaded extends ModelBusyError {}
export class InferenceUnavailable extends ModelBusyError {}
export {SainaError as SainaHelmError, TransportError as SainaConnectionError};
export const ERROR_CLASSES: Readonly<Record<ErrorCode, typeof HTTPError>>;

/** Billing metadata from response headers; credit fields are null when the server omits them. */
export interface ResponseMetadata {
  requestId: string | null;
  creditsCharged: bigint | null;
  balance: bigint | null;
  priceVersion: string | null;
  replayed: boolean;
  idempotencyKey: string | null;
  attempts: number;
}
export interface CallOptions {
  /** UUIDv7; enables bounded automatic retries that reuse this key and the identical body. */
  idempotencyKey?: string;
}
export interface Balance {
  account_id: string; balance: bigint; reserved: bigint; available: bigint; overdraft_allowance: bigint; debt: bigint;
  suspensions: unknown[]; [key: string]: unknown;
}
export type Filter = string | number | Date | null | undefined;
export interface ClientOptions {
  /** Defaults to https://api.saina.run. */
  baseUrl?: string;
  apiKey?: string;
  allowHttp?: boolean;
  /** Per-attempt timeout in ms. */
  timeout?: number;
  fetch?: typeof fetch;
  /** Retries after the first attempt (default 2). */
  maxRetries?: number;
  /** Upper bound on the total wait across retries, in ms (default 30000). */
  maxRetryDelay?: number;
  /** Generate one UUIDv7 key per inference call when none is passed (default true). */
  autoIdempotencyKey?: boolean;
  /** Testing hooks. */
  sleep?: (ms: number) => Promise<void>;
  random?: () => number;
}
export class SainaHelm {
  constructor(options?: ClientOptions);
  readonly baseUrl: string;
  lastMetadata: ResponseMetadata | null;
  ask(request: AskRequest, options?: CallOptions): Promise<AskResponse>;
  askWithMetadata(request: AskRequest, options?: CallOptions): Promise<{result: AskResponse; metadata: ResponseMetadata}>;
  decisions(request: DecisionsRequest, options?: CallOptions): Promise<DecisionsResponse>;
  decisionsWithMetadata(request: DecisionsRequest, options?: CallOptions): Promise<{result: DecisionsResponse; metadata: ResponseMetadata}>;
  systemOne(request: SystemOneRequest, options?: CallOptions): Promise<SystemOneResponse>;
  systemOneWithMetadata(request: SystemOneRequest, options?: CallOptions): Promise<{result: SystemOneResponse; metadata: ResponseMetadata}>;
  balance(): Promise<Balance>;
  usage(filters?: {from?: Filter; to?: Filter; channel?: Filter}): Promise<any>;
  requests(filters?: {cursor?: Filter; limit?: Filter; from?: Filter; to?: Filter; key_id?: Filter; channel?: Filter; status?: Filter}): Promise<any>;
}
export { SainaHelm as Saina };

export interface DecisionsRequest {
  model: string;
  input: Content;
  questions: Array<{name?: string; type: 'predicate' | 'choice' | 'score'; extra?: Record<string, Json>; [key: string]: unknown}>;
  extra?: Record<string, Json>;
  [key: string]: unknown;
}
export interface DecisionsResponse {
  model: string;
  answers: Array<{name: string | null; type: 'predicate' | 'choice' | 'score' | 'refusal'; [key: string]: unknown}>;
}
export function stringifyCredits(value: unknown): string;

export {TraceClient, DecisionTrace, TraceDeliveryError} from './traces.js';
export interface TraceFlushResult {readonly complete: boolean; readonly pending: number; readonly rejected: number}
export interface TracesOptions extends Pick<ClientOptions, 'baseUrl' | 'apiKey' | 'timeout' | 'allowHttp' | 'fetch'> {spoolPath?: string; capacity?: number; onStorageError?: 'warn' | 'raise'}
export class Traces {
  constructor(options?: TracesOptions);
  readonly spoolPath: string | undefined;
  readonly capacity: number;
  readonly pending: Promise<number>;
  readonly lastError: Error | null;
  log(record: {task: string; inputs: Json; outcome?: Json; metadata?: Record<string, Json> | null}): Promise<string | null>;
  flush(options?: {timeout?: number}): Promise<TraceFlushResult>;
  close(options?: {timeout?: number}): Promise<void>;
  exportRejected(path: string): Promise<number>;
  retryRejected(traceIds: string[]): Promise<number>;
  discardRejected(traceIds: string[]): Promise<number>;
}

export type {TraceJson, TraceOptions, DecisionTraceOptions, EndpointDecision} from './traces.js';
