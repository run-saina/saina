import test from 'node:test';
import assert from 'node:assert/strict';
import * as sdk from './index.js';

const request = {model: 'm', state: 's', questions: {q: {type: 'yes_no', question: 'Q?'}}};
const response = {request_id:'req_1', model: 'm', answers: {q: {type: 'yes_no', yes: .9, no: .1, confidence: .8}}};
function client(outcomes, options = {}) {
  const calls = [];
  const instance = new sdk.Saina({apiKey: 'test', sleep: async () => {}, ...options,
    fetch: async (url, init) => {
      calls.push({url, ...init});
      const [body, status = 200, headers = {}] = outcomes.shift();
      return new Response(JSON.stringify(body), {status, headers});
    }});
  return {instance, calls};
}

test('exact aliases and broad HTTP fallbacks, including unknown codes', async () => {
  assert.equal(sdk.SainaHelmError, sdk.SainaError);
  assert.equal(sdk.SainaConnectionError, sdk.TransportError);
  for (const [status, Cls] of [[401, sdk.AuthenticationError], [403, sdk.PermissionDeniedError],
    [404, sdk.NotFoundError], [422, sdk.InvalidRequestError], [429, sdk.RateLimitError],
    [503, sdk.ModelBusyError], [500, sdk.ServerError], [409, sdk.HTTPError]]) {
    for (const body of [{detail: 'legacy'}, {error: {code: 'future_code', custom: 3}}]) {
      const {instance} = client([[body, status]], {autoIdempotencyKey: false});
      await assert.rejects(instance.ask(request), e => e.constructor === Cls);
    }
  }
});

test('recognized code takes priority and preserves future error fields', async () => {
  const fields = {code: 'rate_limited', admitted: false, custom: {nested: [3]}};
  const {instance} = client([[{error: fields}, 500]], {maxRetries: 0});
  await assert.rejects(instance.ask(request), e => {
    assert.ok(e instanceof sdk.RateLimited && e instanceof sdk.RateLimitError);
    assert.deepEqual(e.fields, fields);
    assert.deepEqual(e.raw, {error: fields});
    return true;
  });
});

test('all inference entry points generate one default key reused with identical bytes', async () => {
  for (const name of ['ask', 'askWithMetadata', 'systemOne', 'systemOneWithMetadata', 'decisions', 'decisionsWithMetadata']) {
    const isDecisions = name.startsWith('decisions');
    const body = isDecisions ? {model: 'm', input: 's', questions: [{name: 'q', type: 'predicate', question: 'Q?'}]} : request;
    const result = isDecisions ? {model: 'm', answers: [{name: 'q', type: 'predicate', probability: .9}]} : response;
    const {instance, calls} = client([[{error: {code: 'accounting_unavailable'}}, 503], [result]]);
    await instance[name](body);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].body, calls[1].body);
    assert.equal(calls[0].headers['Idempotency-Key'], calls[1].headers['Idempotency-Key']);
    assert.match(calls[0].headers['Idempotency-Key'], /^[0-9a-f-]{14}7/);
    if (isDecisions) assert.match(calls[0].url, /\/v1\/decisions$/);
  }
});

test('extensions flatten and options lists normalize without changing input', async () => {
  const body = {model: 'm', state: 's', extra: {future: true}, questions: {q: {
    type: 'single_choice', question: 'Q?', options: ['a', 'b'], extra: {future: 3}}}};
  const original = structuredClone(body);
  const {instance, calls} = client([[{request_id:'req_1', model: 'm', answers: {q: {type: 'single_choice', selection: 'a',
    probabilities: {a: .9, b: .1}, confidence: .8}}}]]);
  await instance.ask(body);
  const sent = JSON.parse(calls[0].body);
  assert.equal(sent.future, true);
  assert.equal(sent.questions.q.future, 3);
  assert.deepEqual(sent.questions.q.options, {a: null, b: null});
  assert.ok(!('extra' in sent) && !('extra' in sent.questions.q));
  assert.deepEqual(body, original);
});

test('extension collisions fail before fetch', async () => {
  for (const field of ['type', 'options', 'threshold', 'min_margin', 'descriptions', 'extra']) {
    const {instance, calls} = client([]);
    const body = structuredClone(request);
    body.questions.q.extra = {[field]: null};
    await assert.rejects(instance.ask(body), TypeError);
    assert.equal(calls.length, 0);
  }
  for (const field of ['model', 'questions', 'images', 'mode', 'extra']) {
    const {instance, calls} = client([]);
    await assert.rejects(instance.ask({...request, extra: {[field]: null}}), TypeError);
    assert.equal(calls.length, 0);
  }
});

test('BigInt serialization preserves decimal strings and raw wire response', async () => {
  const big = '18446744073709551615';
  const {instance} = client([[{...response, billing: {balance: big}}, 200, {'X-Saina-Balance': big}]]);
  const result = await instance.ask(request);
  assert.equal(result.metadata.balance, BigInt(big));
  assert.equal(result.raw.billing.balance, big);
  assert.equal(JSON.parse(JSON.stringify(result)).metadata.balance, big);
  assert.equal(JSON.parse(JSON.stringify(result)).metadata.balance, big);
  assert.equal(result.toObject().metadata.balance, big);
  assert.equal(sdk.stringifyCredits({balance: BigInt(big)}), `{"balance":"${big}"}`);
});
