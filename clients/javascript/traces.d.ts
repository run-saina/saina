export type TraceJson = null | boolean | number | string | TraceJson[] | {[key: string]: TraceJson};
export interface TraceOptions {
  collection: string; apiKey?: string; baseUrl?: string; source?: string;
  batchSize?: number; flushInterval?: number; maxPending?: number; timeoutMs?: number;
  maxRetries?: number; onError?: (error: Error) => void;
}
export interface DecisionTraceOptions {
  task: string; inputs: TraceJson; decisionId?: string; metadata?: Record<string, TraceJson>;
}
export class TraceDeliveryError extends Error { status?: number; code?: string; }
export class DecisionTrace {
  readonly decisionId: string;
  setAnswer(answer: TraceJson): void;
}
export interface EndpointDecision {
  decision_id: string; collection_id: string; endpoint: string; endpoint_version: number;
  status: 'completed' | 'error'; state: TraceJson; answers: Record<string, TraceJson>;
  [key: string]: unknown;
}
export class TraceClient {
  constructor(options: TraceOptions);
  readonly collection: string; readonly source: string;
  readonly pendingCount: number; readonly droppedCount: number; readonly lastError: Error | null;
  decision<T>(options: DecisionTraceOptions, fn: (trace: DecisionTrace) => T | Promise<T>): Promise<T>;
  flush(): Promise<void>;
  close(): Promise<void>;
  reportOutcome(decisionId: string, outcome: {values: Record<string, TraceJson>; eventId: string}): Promise<Record<string, unknown>>;
  invokeEndpoint(endpoint: string, request: {state?: TraceJson; questions?: Record<string, TraceJson>; images?: TraceJson[]; audios?: TraceJson[]}, options?: {version?: 'production' | number}): Promise<EndpointDecision>;
}
