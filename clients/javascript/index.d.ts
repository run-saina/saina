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
export class SainaHelmError extends Error { readonly status?: number; }
export class SainaHelm {
  constructor(options: {baseUrl: string; apiKey: string; timeout?: number; fetch?: typeof fetch});
  ask(request: AskRequest): Promise<AskResponse>;
}
export { SainaHelm as Saina, SainaHelmError as SainaError };
