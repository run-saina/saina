/** Immutable answer values and synchronous local selection. */
import {ProtocolError} from './index.js';
import {optionLabels} from './validation.js';
const labelOrder = Symbol('optionOrder');

export function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const item of Object.values(value)) deepFreeze(item);
    Object.freeze(value);
  }
  return value;
}
export const toObject = value => JSON.parse(JSON.stringify(value, (_key, item) => typeof item === 'bigint' ? item.toString() : item));
const number = (v, name) => {
  if (typeof v !== 'number') throw new TypeError(`${name} must be a number`);
  if (!Number.isFinite(v) || v < 0 || v > 1) throw new RangeError(`${name} must be finite and in [0, 1]`);
};

class Serializable {
  toObject() { return toObject({...this}); }
  toJSON() { return this.toObject(); }
}
class Answer extends Serializable {
  select({threshold, minMargin = 0} = {}) {
    number(threshold, 'threshold'); number(minMargin, 'minMargin');
    if (this.type === 'multi_choice') {
      const value = this[labelOrder].filter(k => this.memberships[k] >= threshold);
      return new MultiChoiceSelection(value, value.length > 0 ? 'accepted' : 'no_options_selected', this, threshold, 0);
    }
    const vector = this.type === 'yes_no' ? {yes: this.yes, no: this.no} : this.probabilities;
    const labels = Object.keys(vector).sort((a, b) => vector[b] - vector[a]);
    const [first, second] = labels;
    const lead = vector[first] - vector[second];
    const margin = this.type === 'rating' ? 0 : minMargin;
    const reason = vector[first] < threshold ? 'below_threshold' : lead <= 1e-12 ? 'tie'
      : lead < margin ? 'below_margin' : 'accepted';
    const value = reason !== 'accepted' ? null : this.type === 'yes_no' ? first === 'yes'
      : this.type === 'rating' ? Number(first) : first;
    const Cls = this.type === 'yes_no' ? YesNoSelection : this.type === 'rating' ? RatingSelection : SingleChoiceSelection;
    return new Cls(value, reason, this, threshold, margin);
  }
}
export class YesNoAnswer extends Answer {
  constructor(a) { super(); Object.assign(this, {type: 'yes_no', yes: a.yes, no: a.no, confidence: a.confidence}); deepFreeze(this); }
}
export class SingleChoiceAnswer extends Answer {
  constructor(a) { super(); Object.assign(this, {type: 'single_choice', probabilities: structuredClone(a.probabilities), confidence: a.confidence}); deepFreeze(this); }
}
export class RatingAnswer extends Answer {
  constructor(a) { super(); Object.assign(this, {type: 'rating', probabilities: structuredClone(a.probabilities), levels: structuredClone(a.levels), expected_level: a.expected_level, confidence: a.confidence}); deepFreeze(this); }
}
export class MultiChoiceAnswer extends Answer {
  constructor(a, order = Object.keys(a.memberships)) {
    super();
    Object.defineProperty(this, labelOrder, {value: Object.freeze([...order])});
    Object.assign(this, {type: 'multi_choice', memberships: structuredClone(a.memberships)});
    deepFreeze(this);
  }
}
class Selection extends Serializable {
  constructor(value, reason, answer, threshold, minMargin) {
    super();
    Object.assign(this, {type: answer.type, value, accepted: reason === 'accepted', reason, answer, threshold, minMargin});
    if (answer.type === 'rating') this.label = value === null ? null : answer.levels[value];
    deepFreeze(this);
  }
}
export class YesNoSelection extends Selection {}
export class SingleChoiceSelection extends Selection {}
export class RatingSelection extends Selection {}
export class MultiChoiceSelection extends Selection {}

export class Selections extends Serializable {
  constructor(items) {
    super();
    for (const [key, value] of Object.entries(items)) Object.defineProperty(this, key, {value, enumerable: true});
    deepFreeze(this);
  }
}

export class Answers extends Serializable {
  #raw;
  constructor(raw, metadata, questions) {
    super();
    this.#raw = structuredClone(raw);
    const classes = {yes_no: YesNoAnswer, single_choice: SingleChoiceAnswer, rating: RatingAnswer, multi_choice: MultiChoiceAnswer};
    Object.assign(this, {requestId: metadata.requestId, model: raw.model, specialist: raw.specialist ?? null,
      usage: structuredClone(raw.usage ?? {}), metadata: {...metadata},
      answers: Object.fromEntries(Object.entries(questions).map(([key, q]) => {
        const a = {...raw.answers[key]};
        const field = a.type === 'multi_choice' ? 'memberships' : a.type === 'single_choice' ? 'probabilities' : null;
        if (field) a[field] = Object.fromEntries(Object.keys(q.options).map(k => [k, a[field][k]]));
        return [key, new classes[a.type](a, optionLabels.get(q))];
      }))});
    deepFreeze(this);
  }
  get raw() { return structuredClone(this.#raw); }
  select({threshold, minMargin = 0} = {}) {
    number(threshold, 'threshold'); number(minMargin, 'minMargin');
    return new Selections(Object.fromEntries(Object.entries(this.answers).map(([key, a]) => [key, a.select({threshold, minMargin})])));
  }
}

export function validateTyped(request, result) {
  const fail = () => { throw new ProtocolError('Invalid typed answer response'); };
  const p = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1;
  const near = (v, expected) => typeof v === 'number' && Number.isFinite(v) && Math.abs(v - expected) <= 1e-9;
  if (!result || typeof result.model !== 'string' || !result.model || !result.answers || Array.isArray(result.answers) ||
      Object.keys(result.answers).length !== Object.keys(request.questions).length ||
      result.specialist != null && typeof result.specialist !== 'string') fail();
  if (result.usage != null && (typeof result.usage !== 'object' || Array.isArray(result.usage) ||
      Object.entries(result.usage).some(([k, v]) => ['input_tokens', 'output_tokens'].includes(k) && (!Number.isInteger(v) || v < 0)))) fail();
  for (const [key, q] of Object.entries(request.questions)) {
    const a = Object.hasOwn(result.answers, key) && result.answers[key];
    if (!a || a.type !== q.type) fail();
    const labels = q.type === 'yes_no' ? ['yes', 'no'] : q.type === 'rating' ? q.levels.map((_, i) => String(i)) : Object.keys(q.options);
    const vector = q.type === 'yes_no' ? {yes: a.yes, no: a.no} : q.type === 'multi_choice' ? a.memberships : a.probabilities;
    if (!vector || Array.isArray(vector) || Object.keys(vector).length !== labels.length || !labels.every(k => Object.hasOwn(vector, k) && p(vector[k]))) fail();
    if (q.type === 'multi_choice') continue;
    const values = labels.map(k => vector[k]);
    if (Math.abs(values.reduce((s, v) => s + v, 0) - 1) > 1e-5) fail();
    const n = values.length, highest = Math.max(...values);
    let confidence = q.type === 'yes_no' ? 2 * highest - 1 : (highest - 1/n)/(1 - 1/n);
    if (q.type === 'rating') {
      const peak = values.indexOf(highest);
      const mean = values.reduce((s, _v, i) => s + Math.abs(i - (n - 1)/2), 0)/n;
      confidence = 1 - values.reduce((s, v, i) => s + v * Math.abs(i - peak), 0)/mean;
      if (JSON.stringify(a.levels) !== JSON.stringify(q.levels) || !near(a.expected_level, values.reduce((s, v, i) => s + v*i, 0))) fail();
    }
    if (!p(a.confidence) || !near(a.confidence, Math.max(0, Math.min(1, confidence)))) fail();
  }
}
