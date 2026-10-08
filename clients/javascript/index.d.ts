export type Content = string | {[key: string]: Json} | Json[];
export type Json = null | boolean | number | string | Json[] | {[key: string]: Json};
export type Options = Record<string, Content | null>;
export type Question =
  | {type: 'yes_no'; question: Content; descriptions?: {yes: Content; no: Content}}
  | {type: 'single_choice'; question: Content; options: Options}
  | {type: 'rating'; question: Content; levels: Content[]}
  | {type: 'multi_choice'; question: Content; options: Options};
export type DecisionQuestion = Question extends infer Q ? Q extends Question ? Q & {threshold?: number} &
  (Q['type'] extends 'yes_no' | 'single_choice' ? {min_margin?: number} : {}) : never : never;
export type AskRequest = {model: 'helm-0.8b' | 'saina-helm-0.8b' | 'saina-helm'; state: Content} & (
  | {mode?: 'distribution'; questions: Record<string, Question>}
  | {mode: 'decision'; threshold?: number; min_margin?: number; questions: Record<string, DecisionQuestion>});
export type Reason = 'accepted' | 'below_threshold' | 'below_margin' | 'tie';
export type Answer =
  | {type: 'yes_no'; yes: number; no: number; confidence: number; selected?: boolean | null; reason?: Reason}
  | {type: 'single_choice'; selection: string | null; probabilities: Record<string, number>; confidence: number; reason?: Reason}
  | {type: 'rating'; expected_level: number; levels: Content[]; probabilities: Record<string, number>; confidence: number; level?: number | null; reason?: Reason}
  | {type: 'multi_choice'; memberships: Record<string, number>; selections?: string[]; reason?: Reason};
export interface AskResponse {model: string; answers: Record<string, Answer>; usage: {input_tokens: number; output_tokens: number}}
export type SystemOneQuestion = {type: 'choice' | 'score' | 'noul'; instructions: Content; criteria?: {[key: string]: Json} | Json[] | null};
export type SystemOneRequest = {model: string; state: Content; questions: Record<string, SystemOneQuestion>};
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

export type ErrorCode =
  | 'invalid_api_key' | 'key_revoked' | 'account_suspended' | 'insufficient_credits' | 'invalid_request'
  | 'rate_limited' | 'overloaded' | 'inference_failed' | 'inference_unavailable' | 'accounting_unavailable'
  | 'idempotency_conflict' | 'request_in_progress' | 'idempotency_result_expired' | 'idempotency_unverifiable'
  | 'fresh_auth_required' | 'permission_denied' | 'not_found';
export interface Suspension {types: Array<'financial' | 'security'>; next_step: 'buy_credits' | 'contact_support'}
export class SainaHelmError extends Error {
  readonly status?: number;
  readonly code: ErrorCode | string | null;
  readonly requestId: string | null;
  readonly admitted: boolean | null;
  readonly state: 'not_admitted' | 'in_progress' | 'terminal' | 'unknown' | null;
  readonly retry: 'same_operation' | 'new_operation' | 'no' | null;
  /** Seconds from Retry-After. */
  readonly retryAfter: number | null;
  readonly suspension: Suspension | null;
}
export class SainaConnectionError extends SainaHelmError {}
export const ERROR_CLASSES: Readonly<Record<ErrorCode, typeof SainaHelmError>>;
export class InvalidApiKey extends SainaHelmError {}
export class KeyRevoked extends SainaHelmError {}
export class AccountSuspended extends SainaHelmError {}
export class InsufficientCredits extends SainaHelmError {}
export class InvalidRequest extends SainaHelmError {}
export class RateLimited extends SainaHelmError {}
export class Overloaded extends SainaHelmError {}
export class InferenceFailed extends SainaHelmError {}
export class InferenceUnavailable extends SainaHelmError {}
export class AccountingUnavailable extends SainaHelmError {}
export class IdempotencyConflict extends SainaHelmError {}
export class RequestInProgress extends SainaHelmError {}
export class IdempotencyResultExpired extends SainaHelmError {}
export class IdempotencyUnverifiable extends SainaHelmError {}
export class FreshAuthRequired extends SainaHelmError {}
export class PermissionDenied extends SainaHelmError {}
export class NotFound extends SainaHelmError {}

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
  apiKey: string;
  /** Per-attempt timeout in ms. */
  timeout?: number;
  fetch?: typeof fetch;
  /** Retries after the first attempt (default 2). */
  maxRetries?: number;
  /** Upper bound on the total wait across retries, in ms (default 30000). */
  maxRetryDelay?: number;
  /** Generate one UUIDv7 key per inference call when none is passed, enabling retries. */
  autoIdempotencyKey?: boolean;
  /** Testing hooks. */
  sleep?: (ms: number) => Promise<void>;
  random?: () => number;
}
export class SainaHelm {
  constructor(options: ClientOptions);
  readonly baseUrl: string;
  lastMetadata: ResponseMetadata | null;
  ask(request: AskRequest, options?: CallOptions): Promise<AskResponse>;
  askWithMetadata(request: AskRequest, options?: CallOptions): Promise<{result: AskResponse; metadata: ResponseMetadata}>;
  systemOne(request: SystemOneRequest, options?: CallOptions): Promise<SystemOneResponse>;
  systemOneWithMetadata(request: SystemOneRequest, options?: CallOptions): Promise<{result: SystemOneResponse; metadata: ResponseMetadata}>;
  balance(): Promise<Balance>;
  usage(filters?: {from?: Filter; to?: Filter; channel?: Filter}): Promise<any>;
  requests(filters?: {cursor?: Filter; limit?: Filter; from?: Filter; to?: Filter; key_id?: Filter; channel?: Filter; status?: Filter}): Promise<any>;
}
export { SainaHelm as Saina, SainaHelmError as SainaError };
