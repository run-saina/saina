import test from 'node:test';
import assert from 'node:assert/strict';
import {Saina, ProtocolError, TransportError, YesNoAnswer, RatingAnswer, MultiChoiceAnswer, SingleChoiceAnswer} from './index.js';
const request = {model:'m',state:'s',questions:{q:{type:'yes_no',question:'Q?'}}};
const raw = {request_id:'r',model:'m',answers:{q:{type:'yes_no',yes:.9,no:.1,confidence:.8}}};
const client = (result = raw, headers = {}) => {
  const calls = [];
  return {calls, sdk: new Saina({apiKey:'key',fetch:async (url, init)=>{
    calls.push({url,...init}); return new Response(JSON.stringify(result), {headers});
  }})};
};

test('local selection obeys threshold, tie, margin, and zero/false acceptance', () => {
  const cases = [
    [new YesNoAnswer({yes:.5,no:.5,confidence:0}), .9, 0, null, 'below_threshold'],
    [new YesNoAnswer({yes:.5,no:.5,confidence:0}), .4, 0, null, 'tie'],
    [new YesNoAnswer({yes:.1,no:.9,confidence:.8}), .9, .8, false, 'accepted'],
    [new SingleChoiceAnswer({probabilities:{a:.6,b:.4},confidence:.2}), .5, .3, null, 'below_margin'],
    [new RatingAnswer({probabilities:{0:.9,1:.1},levels:['low','high'],expected_level:.1,confidence:.8}), .9, 1, 0, 'accepted'],
    [new RatingAnswer({probabilities:{0:.5,1:.5},levels:['same','same'],expected_level:.5,confidence:0}), .9, 1, null, 'below_threshold'],
    [new RatingAnswer({probabilities:{0:.5,1:.5},levels:['same','same'],expected_level:.5,confidence:0}), .4, 1, null, 'tie'],
    [new MultiChoiceAnswer({memberships:{b:.9,a:.8,c:.1}}), .8, 1, ['b','a'], 'accepted'],
    [new MultiChoiceAnswer({memberships:{a:.1,b:.2}}), .9, 0, [], 'no_options_selected'],
  ];
  for (const [a, threshold, minMargin, value, reason] of cases) {
    const s = a.select({threshold,minMargin});
    assert.deepEqual(s.value,value); assert.equal(s.reason,reason); assert.equal(s.accepted,reason==='accepted');
    assert.equal(s.answer,a); assert.equal(s.minMargin,['rating','multi_choice'].includes(a.type)?0:minMargin);
    assert.equal(s.toObject().type,a.type);
    assert.deepEqual(JSON.parse(JSON.stringify(s)),s.toObject());
    if (a.type==='rating') assert.equal(s.label,value===null?null:a.levels[value]);
  }
});

test('selection arguments validate in individual and bulk calls', async () => {
  const {sdk} = client(); const result = await sdk.ask(request);
  for (const bad of [NaN, Infinity, -.1, 1.1, true, null, '0.9', undefined]) {
    assert.throws(()=>result.select({threshold:bad}));
    assert.throws(()=>result.answers.q.select({threshold:bad}));
    assert.throws(()=>result.select({threshold:.9,minMargin:bad === undefined ? null : bad}));
  }
  assert.throws(()=>result.select());
});

test('typed results, nested values, metadata and raw snapshots are independent', async () => {
  const body = {...raw, future:{credits:'18446744073709551615'}};
  const {sdk, calls} = client(body, {'X-Saina-Balance':'18446744073709551615'});
  const result = await sdk.ask(request);
  assert.equal(result.requestId,'r');
  assert.equal(result.metadata.balance,18446744073709551615n);
  assert.equal(result.select({threshold:.9}).q.value,true);
  assert.equal(calls.length,1);
  assert.throws(()=>{result.answers.q.yes=0;},TypeError);
  assert.throws(()=>{result.metadata.balance=0n;},TypeError);
  assert.throws(()=>{result.answers.x=null;},TypeError);
  const snapshot = result.raw; snapshot.future.credits='0';
  assert.equal(result.raw.future.credits,'18446744073709551615');
  const copy = result.toObject(); copy.answers.q.yes=0;
  assert.equal(result.answers.q.yes,.9);
  assert.equal(JSON.parse(JSON.stringify(result)).metadata.balance,'18446744073709551615');
  assert.deepEqual(result.select({threshold:.9}).toObject().q.answer,result.answers.q.toObject());
});

test('bulk and individual selections preserve declared option order', async () => {
  const q={model:'m',state:'s',questions:{q:{type:'multi_choice',question:'Q?',options:['b','a']}}};
  const {sdk} = client({...raw,answers:{q:{type:'multi_choice',memberships:{a:.9,b:.8}}}});
  const result=await sdk.ask(q);
  assert.deepEqual(result.select({threshold:.8,minMargin:1}).q.value,['b','a']);
  assert.equal(result.select({threshold:.8,minMargin:1}).q.minMargin,0);
});

test('request ID must be present and consistent', async () => {
  for (const [bodyId, headerId, valid] of [[null,null,false],['body',null,true],[null,'header',true],
    ['same','same',true],['body','header',false],['',null,false],[3,null,false]]) {
    const body=structuredClone(raw); delete body.request_id;
    if(bodyId!==null) body.request_id=bodyId;
    const {sdk,calls}=client(body,headerId===null?{}:{'X-Request-Id':headerId});
    if(valid) assert.equal((await sdk.ask(request)).requestId,bodyId??headerId);
    else await assert.rejects(sdk.ask(request),ProtocolError);
    assert.equal(calls.length,1);
  }
});

test('derived fields and rating echoes must match the wire formulas', async () => {
  for (const patch of [{confidence:.5},{yes:.8},{no:null}]) {
    const body=structuredClone(raw); Object.assign(body.answers.q,patch);
    const {sdk,calls}=client(body);
    await assert.rejects(sdk.ask(request),ProtocolError); assert.equal(calls.length,1);
  }
  const q={model:'m',state:'s',questions:{q:{type:'rating',question:'Q?',levels:['a','b','c']}}};
  const body={...raw,answers:{q:{type:'rating',probabilities:{0:.1,1:.2,2:.7},levels:['a','b','c'],expected_level:1.6,confidence:.4}}};
  assert.equal((await client(body).sdk.ask(q)).answers.q.select({threshold:.7}).value,2);
  for(const patch of [{confidence:.5},{expected_level:1.5},{levels:['a','b','wrong']}]) {
    const bad=structuredClone(body); Object.assign(bad.answers.q,patch);
    await assert.rejects(client(bad).sdk.ask(q),ProtocolError);
  }
});

test('invalid requests fail before media downloads', async () => {
  const cyclic=[];cyclic.push(cyclic);
  const changes=[{state:{n:NaN}},{state:undefined},{state:cyclic},{extra:{idempotencyKey:'override'}},
    {questions:{q:{type:'single_choice',question:'Q?',options:['a','a']}}},
    {questions:{q:{type:'rating',question:'Q?',levels:['one']}}},
    {questions:{q:{type:'yes_no',question:'Q?',typo:true}}}];
  for(const change of changes) {
    const {sdk,calls}=client();
    await assert.rejects(sdk.ask({...request,images:['https://example.com/image'],...change}));
    assert.equal(calls.length,0);
  }
});

test('environment configuration resolves at construction and HTTP requires opt-in', () => {
  const previous={...process.env};
  try {
    process.env.SAINA_API_KEY='env-key';process.env.SAINA_BASE_URL='https://env.example';process.env.SAINA_TIMEOUT='12.5';
    const sdk=new Saina();assert.equal(sdk.timeout,12500);assert.equal(sdk.baseUrl,'https://env.example');
    assert.equal(new Saina({baseUrl:'https://explicit.example',apiKey:'key',timeout:3}).timeout,3);
    assert.throws(()=>new Saina({baseUrl:'http://saina:8700'}));
    assert.equal(new Saina({baseUrl:'http://saina:8700',allowHttp:true}).baseUrl,'http://saina:8700');
    process.env.SAINA_BASE_URL='https://new.example';assert.equal(new Saina().baseUrl,'https://new.example');
    assert.equal(sdk.baseUrl,'https://env.example');
  } finally {
    for(const key of ['SAINA_API_KEY','SAINA_BASE_URL','SAINA_TIMEOUT']) {
      if(previous[key]===undefined) delete process.env[key];else process.env[key]=previous[key];
    }
  }
});

test('malformed JSON is a protocol failure without retry and timeouts expose timedOut', async () => {
  let attempts=0;
  const sdk=new Saina({apiKey:'key',fetch:async()=>{attempts++;return new Response('not json');}});
  await assert.rejects(sdk.ask(request),ProtocolError);assert.equal(attempts,1);
  const timed=new Saina({apiKey:'key',maxRetries:0,fetch:async()=>{throw new DOMException('timed out','TimeoutError');}});
  await assert.rejects(timed.ask(request),e=>e instanceof TransportError && e.timedOut);
});

test('invalid keys fail before media and numeric option labels retain list order', async () => {
  const {sdk,calls}=client();
  for(const idempotencyKey of ['bad','01890a5d-ac96-774b-bcce-b302099a8057\n'])
    await assert.rejects(sdk.ask({...request,images:['https://example.com/image']},{idempotencyKey}),TypeError);
  assert.equal(calls.length,0);
  const q={model:'m',state:'s',questions:{q:{type:'multi_choice',question:'Q?',options:['2','1']}}};
  const result=await client({...raw,answers:{q:{type:'multi_choice',memberships:{1:.9,2:.8}}}}).sdk.ask(q);
  assert.deepEqual(result.answers.q.select({threshold:.8}).value,['2','1']);
});
