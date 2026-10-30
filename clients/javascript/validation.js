/** Validate JSON without silently dropping undefined, unsupported values, or non-finite numbers. */
export function jsonValue(value, active = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new RangeError('JSON numbers must be finite');
    return;
  }
  if (!value || typeof value !== 'object' || ![Object.prototype, null, Array.prototype].includes(Object.getPrototypeOf(value)))
    throw new TypeError('Unsupported JSON value');
  if (active.has(value)) throw new TypeError('Cyclic JSON value');
  active.add(value);
  try {
    if (Object.getOwnPropertySymbols(value).length) throw new TypeError('JSON object keys must be strings');
    if (Array.isArray(value)) { for (const item of value) jsonValue(item, active); }
    else { for (const item of Object.values(value)) jsonValue(item, active); }
  } finally { active.delete(value); }
}
export const questionFields = new Set(['type', 'question', 'options', 'levels', 'instructions', 'criteria', 'descriptions', 'name', 'choices', 'threshold', 'min_margin', 'extra']);
export const requestFields = new Set(['model', 'state', 'input', 'questions', 'mode', 'images', 'audios', 'videos', 'specialist', 'threshold', 'min_margin', 'extra', 'idempotency_key', 'idempotencyKey', 'task']);
// A task names the call in Saina Studio, which keeps it under an automatic endpoint of the same name.
export function taskName(value) {
  if (value != null && !(typeof value === 'string' && /^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$/.test(value)))
    throw new TypeError('task must be 3-64 lowercase letters, numbers or hyphens, starting and ending with a letter or number');
  return value;
}
export function flattenExtra(value, known) {
  const {extra, ...result} = value;
  if (extra == null) return result;
  if (typeof extra !== 'object' || Array.isArray(extra)) throw new TypeError('extra must be an object');
  jsonValue(extra);
  for (const key of Object.keys(extra)) {
    if (known.has(key) || Object.hasOwn(result, key)) throw new TypeError('extra cannot override known fields');
  }
  return {...result, ...extra};
}
const label = (v, name) => {
  if (typeof v !== 'string') throw new TypeError(`${name} must be a string`);
  if (!v.trim()) throw new TypeError(`${name} must not be blank`);
};
export const optionLabels = new WeakMap();
export function stringifyRequest(request) {
  return JSON.stringify(request, function (key, value) {
    const order = key === 'options' && optionLabels.get(this);
    // Ordinary objects enumerate integer-like keys numerically. A proxy retains
    // the caller's list order in JSON without changing the native wire schema.
    return order ? new Proxy(value, {ownKeys: () => order}) : value;
  });
}
const content = (value, name) => {
  if (typeof value !== 'string' && (!value || typeof value !== 'object')) throw new TypeError(`${name} must be text or structured content`);
  jsonValue(value);
};
export function nativeRequest(request) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) throw new TypeError('request must be an object');
  const allowed = new Set(['model', 'state', 'questions', 'images', 'audios', 'specialist', 'extra', 'task']);
  if (Object.keys(request).some(k => !allowed.has(k))) throw new TypeError('Unknown request fields; use extra and local select()');
  label(request.model, 'model');
  if (request.model.length > 64) throw new TypeError('model exceeds 64 characters');
  if (!Object.hasOwn(request, 'state')) throw new TypeError('state is required');
  content(request.state, 'state');
  if (request.specialist != null) label(request.specialist, 'specialist');
  taskName(request.task);
  const questions = request.questions;
  if (!questions || typeof questions !== 'object' || Array.isArray(questions)) throw new TypeError('questions must be an object');
  const entries = Object.entries(questions);
  if (!entries.length || entries.length > 256) throw new TypeError('Expected 1 to 256 questions');
  const normalized = entries.map(([id, q]) => {
    label(id, 'Question ID');
    if (!q || typeof q !== 'object' || Array.isArray(q)) throw new TypeError('question must be an object');
    if (!['yes_no', 'single_choice', 'multi_choice', 'rating'].includes(q.type)) throw new TypeError('Unsupported question type');
    const known = new Set(['type', 'question', 'extra', ...(q.type === 'yes_no' ? ['descriptions'] : q.type === 'rating' ? ['levels'] : ['options'])]);
    if (Object.keys(q).some(k => !known.has(k))) throw new TypeError('Unknown question fields; use extra and local select()');
    if (!Object.hasOwn(q, 'question')) throw new TypeError('question is required');
    content(q.question, 'question');
    q = {...q};
    let order;
    if (['single_choice', 'multi_choice'].includes(q.type)) {
      if (!q.options || typeof q.options !== 'object') throw new TypeError('options must be an object or list');
      const labels = Array.isArray(q.options) ? q.options : Object.keys(q.options);
      if (labels.length < 2 || labels.length > 255) throw new TypeError('Expected 2 to 255 options');
      labels.forEach(v => label(v, 'Option label'));
      order = [...labels];
      if (new Set(labels).size !== labels.length) throw new TypeError('Option labels must be unique');
      if (Array.isArray(q.options)) q.options = Object.fromEntries(labels.map(v => [v, null]));
      for (const value of Object.values(q.options)) if (value !== null) content(value, 'Option description');
    } else if (q.type === 'rating') {
      if (!Array.isArray(q.levels) || q.levels.length < 2 || q.levels.length > 10) throw new TypeError('Expected 2 to 10 levels');
      q.levels.forEach(v => content(v, 'Level description'));
    } else if (q.descriptions != null) {
      if (typeof q.descriptions !== 'object' || Object.keys(q.descriptions).length !== 2 ||
          !['yes', 'no'].every(k => Object.hasOwn(q.descriptions, k)))
        throw new TypeError('descriptions must contain yes and no');
      Object.values(q.descriptions).forEach(v => content(v, 'Description'));
    }
    q = flattenExtra(q, questionFields);
    jsonValue(q);
    q = structuredClone(q);
    if (order) optionLabels.set(q, order);
    return [id, q];
  });
  const result = {...flattenExtra(request, requestFields), questions: Object.fromEntries(normalized),
    state: structuredClone(request.state), mode: 'distribution'};
  const {images, audios, ...json} = result;
  jsonValue(json);
  return result;
}
export function normalizeRequest(request) {
  taskName(request.task);
  const result = flattenExtra(request, requestFields);
  result.questions = Object.fromEntries(Object.entries(result.questions).map(([key, value]) => [key, flattenExtra(value, questionFields)]));
  return result;
}
