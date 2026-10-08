import test from 'node:test';
import assert from 'node:assert/strict';
import {SainaHelm, Saina, SainaError, SainaConnectionError, newIdempotencyKey, parseCredits, DEFAULT_BASE_URL,
  InsufficientCredits, InferenceFailed, IdempotencyConflict, AccountingUnavailable, AccountSuspended, Overloaded,
  RequestInProgress, RateLimited, ERROR_CLASSES} from './index.js';
const request = {model:'saina-helm-0.8b',state:'Text',questions:{tags:{type:'multi_choice',question:'Tags?',options:{a:null,b:null}}}};
const response = {model:'saina-helm-0.8b',answers:{tags:{type:'multi_choice',memberships:{a:.9,b:.8}}},usage:{input_tokens:1,output_tokens:0}};
const KEY = '01890a5d-ac96-774b-bcce-b302099a8057';
const json = (body, status = 200, headers = {}) =>
  new Response(typeof body === 'string' ? body : JSON.stringify(body), {status, headers: {'Content-Type': 'application/json', ...headers}});
const fail = (status, code, fields = {}, headers = {}) =>
  json({error: {code, message: code, request_id: 'req_1', ...fields}, detail: code}, status, headers);

/** A client whose fetch replays scripted outcomes (Response or Error) and records every call. */
function scripted(outcomes, options = {}) {
  const calls = [], sleeps = [];
  const client = new SainaHelm({baseUrl: 'https://example.com', apiKey: 'test', random: () => 1,
    sleep: async ms => { sleeps.push(ms); }, ...options,
    fetch: async (url, init) => {
      calls.push({url, init});
      const next = outcomes.shift();
      if (next instanceof Error) throw next;
      return next;
    }});
  return {client, calls, sleeps};
}

test('ask sends typed questions and accepts independent memberships', async () => {
  const client = new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async (url, init) => {
    assert.equal(url,'https://example.com/v1/ask');
    assert.deepEqual(JSON.parse(init.body),request);
    assert.equal(init.redirect,'error');
    assert.equal(init.headers.Authorization,'Bearer test');
    assert.equal(init.headers['Idempotency-Key'], undefined);
    return json(response);
  }});
  assert.deepEqual(await client.ask(request),response);
});
test('rejects missing answers, wrong tags and invalid probabilities',async () => {
  for (const answers of [{}, {tags:{type:'single_choice',probabilities:{a:.9,b:.1},selection:'a',confidence:.8}}, {tags:{type:'multi_choice',memberships:{a:NaN,b:.8}}}]) {
    const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async()=>json({...response,answers})});
    await assert.rejects(client.ask(request),/Invalid typed/);
  }
});
test('decision mode carries policy and nullable selection',async()=>{
  const body={model:request.model,state:'Text',mode:'decision',threshold:.95,questions:{team:{type:'single_choice',question:'Team?',options:{a:null,b:null}}}};
  const result={...response,answers:{team:{type:'single_choice',selection:null,probabilities:{a:.9,b:.1},confidence:.8,reason:'below_threshold'}}};
  const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async(_,init)=>{
    assert.deepEqual(JSON.parse(init.body),body);return json(result);}});
  assert.deepEqual(await client.ask(body),result);
});
test('sanitizes HTTP failure',async()=>{
  const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async()=>({ok:false,status:503})});
  await assert.rejects(client.ask(request),e=>e.status===503 && e instanceof SainaError);
});
test('rejects unsafe URL',()=>{assert.throws(()=>new SainaHelm({baseUrl:'http://example.com',apiKey:'test'}));});
test('defaults to the hosted API', () => {
  assert.equal(new Saina({apiKey: 'k'}).baseUrl, DEFAULT_BASE_URL);
  assert.equal(DEFAULT_BASE_URL, 'https://api.saina.run');
});

test('newIdempotencyKey is a UUIDv7 with the current timestamp', () => {
  const before = Date.now();
  const keys = new Set(Array.from({length: 100}, newIdempotencyKey));
  const after = Date.now();
  assert.equal(keys.size, 100);
  for (const key of keys) {
    assert.match(key, /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    const ms = parseInt(key.replace(/-/g, '').slice(0, 12), 16);
    assert.ok(before <= ms && ms <= after);
  }
});
test('rejects a non-UUIDv7 key', async () => {
  const {client} = scripted([]);
  await assert.rejects(client.ask(request, {idempotencyKey: crypto.randomUUID()}), TypeError);
});

test('no key means no retry', async () => {
  const {client, calls} = scripted([fail(503, 'accounting_unavailable', {admitted: true, state: 'unknown'}), json(response)]);
  await assert.rejects(client.ask(request), AccountingUnavailable);
  assert.equal(calls.length, 1);
});
test('retries reuse the same key and identical body', async () => {
  const {client, calls, sleeps} = scripted([
    new TypeError('fetch failed'),
    fail(503, 'accounting_unavailable', {admitted: true, state: 'unknown', retry: 'same_operation'}),
    json(response, 200, {'X-Request-Id': 'req_9', 'X-Saina-Credits-Charged': '3',
      'X-Saina-Balance': '9007199254740993', 'X-Saina-Price-Version': '1', 'X-Saina-Replayed': 'true'})]);
  const {result, metadata} = await client.askWithMetadata(request, {idempotencyKey: KEY});
  assert.deepEqual(result, response);
  assert.equal(calls.length, 3);
  assert.deepEqual(new Set(calls.map(c => c.init.headers['Idempotency-Key'])), new Set([KEY]));
  assert.equal(new Set(calls.map(c => c.init.body)).size, 1);
  assert.deepEqual(metadata, {requestId: 'req_9', creditsCharged: 3n, balance: 9007199254740993n, priceVersion: '1',
    replayed: true, idempotencyKey: KEY, attempts: 3});
  assert.deepEqual(client.lastMetadata, metadata);
  assert.deepEqual(sleeps, [500, 1000]);
});
test('autoIdempotencyKey generates one key per logical call', async () => {
  const {client, calls} = scripted([json('<html>', 502), json(response), json(response)], {autoIdempotencyKey: true});
  await client.ask(request);
  await client.ask(request);
  const keys = calls.map(c => c.init.headers['Idempotency-Key']);
  assert.equal(keys.length, 3);
  assert.equal(keys[0], keys[1]);
  assert.notEqual(keys[1], keys[2]);
});
test('terminal errors are not retried', async () => {
  for (const [status, code, fields, Cls] of [
    [502, 'inference_failed', {admitted: true, state: 'terminal', retry: 'new_operation'}, InferenceFailed],
    [409, 'idempotency_conflict', {retry: 'no'}, IdempotencyConflict],
    [402, 'insufficient_credits', {admitted: false, state: 'not_admitted', retry: 'no'}, InsufficientCredits],
    [503, 'overloaded', {admitted: true, retry: 'new_operation'}, Overloaded],
    [410, 'idempotency_result_expired', {retry: 'new_operation'}, ERROR_CLASSES.idempotency_result_expired]]) {
    const {client, calls} = scripted([fail(status, code, fields), json(response)]);
    await assert.rejects(client.ask(request, {idempotencyKey: KEY}), e => e instanceof Cls && e instanceof SainaError && e.status === status);
    assert.equal(calls.length, 1, code);
  }
});
test('overloaded is retried only when not admitted, honoring Retry-After', async () => {
  const {client, calls, sleeps} = scripted([
    fail(429, 'overloaded', {admitted: false, state: 'not_admitted'}, {'Retry-After': '2'}),
    fail(503, 'inference_unavailable', {admitted: false}), json(response)]);
  assert.deepEqual(await client.ask(request, {idempotencyKey: KEY}), response);
  assert.equal(calls.length, 3);
  assert.deepEqual(sleeps, [2000, 1000]);
});
test('retries are bounded by count and total delay', async () => {
  let s = scripted(Array.from({length: 4}, () => fail(409, 'request_in_progress', {state: 'in_progress'}, {'Retry-After': '1'})));
  await assert.rejects(s.client.ask(request, {idempotencyKey: KEY}), RequestInProgress);
  assert.equal(s.calls.length, 3);
  s = scripted([fail(429, 'rate_limited', {admitted: false}, {'Retry-After': '120'}), json(response)]);
  await assert.rejects(s.client.ask(request, {idempotencyKey: KEY}), e => e instanceof RateLimited && e.retryAfter === 120);
  assert.deepEqual(s.sleeps, []);
  s = scripted([new TypeError('x'), new TypeError('x'), new TypeError('x')], {autoIdempotencyKey: true});
  await assert.rejects(s.client.ask(request), SainaConnectionError);
  assert.equal(s.calls.length, 3);
});
test('error fields are parsed; legacy bodies still give SainaError', async () => {
  const suspension = {types: ['financial'], next_step: 'buy_credits'};
  let s = scripted([fail(403, 'account_suspended', {admitted: false, state: 'not_admitted', retry: 'no', suspension})]);
  await assert.rejects(s.client.ask(request), e => {
    assert.ok(e instanceof AccountSuspended);
    assert.deepEqual([e.status, e.code, e.requestId, e.admitted, e.state, e.retry, e.suspension],
      [403, 'account_suspended', 'req_1', false, 'not_admitted', 'no', suspension]);
    return true;
  });
  s = scripted([json({detail: 'Model is busy'}, 503)]);
  await assert.rejects(s.client.ask(request), e => e.constructor === SainaError && e.status === 503 && e.code === null && /busy/.test(e.message));
});
test('systemOne posts to /v1/systemone', async () => {
  const {client, calls} = scripted([json({model: 'm', answers: {a: {type: 'noul', noul: .5}}})]);
  const body = {model: 'm', state: 's', questions: {a: {type: 'noul', instructions: 'Is it?'}}};
  assert.equal((await client.systemOne(body)).answers.a.noul, .5);
  assert.equal(calls[0].url, 'https://example.com/v1/systemone');
});

test('account methods parse credit strings to BigInt beyond 2^53', async () => {
  const big = '18446744073709551615';
  const {client, calls} = scripted([
    json(`{"account_id":"acc","balance":"${big}","reserved":"0","available":"${big}","overdraft_allowance":"0","debt":"-5","suspensions":[]}`),
    json({rows: [{day: '2026-10-01', credits: '9007199254740993', input_tokens: 7}]}),
    json({requests: [{request_id: 'req_1', credits_charged: '12', balance_after: big}], next_cursor: null})]);
  assert.deepEqual(await client.balance(), {account_id: 'acc', balance: 2n ** 64n - 1n, reserved: 0n, available: 2n ** 64n - 1n,
    overdraft_allowance: 0n, debt: -5n, suspensions: []});
  const usage = await client.usage({from: '2026-10-01', to: undefined, channel: 'api'});
  assert.equal(usage.rows[0].credits, 9007199254740993n);
  assert.equal(usage.rows[0].input_tokens, 7);
  const row = (await client.requests({limit: 10, status: 'settled'})).requests[0];
  assert.deepEqual([row.credits_charged, row.balance_after], [12n, 2n ** 64n - 1n]);
  assert.deepEqual(calls.map(c => [c.url, c.init.method]), [
    ['https://example.com/v1/account/balance', 'GET'],
    ['https://example.com/v1/account/usage?from=2026-10-01&channel=api', 'GET'],
    ['https://example.com/v1/account/requests?limit=10&status=settled', 'GET']]);
});
test('parseCredits never rounds numeric credit fields', () => {
  assert.equal(parseCredits('{"balance": 42}').balance, 42n);
  const big = '{"balance": 9007199254740993}';
  // Exact via the reviver source text where supported, otherwise refused rather than rounded.
  try { assert.equal(parseCredits(big).balance, 9007199254740993n); }
  catch (e) { assert.ok(e instanceof SainaError); }
});
