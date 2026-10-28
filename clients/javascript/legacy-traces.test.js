import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {TraceClient, TraceDeliveryError} from './index.js';

async function fixture(t, options = {}) {
  const requests = [], errors = []; let failure = 0, badAck = false;
  const server = createServer(async (req, res) => {
    const chunks=[]; for await (const chunk of req) chunks.push(chunk);
    const body=JSON.parse(Buffer.concat(chunks));requests.push({path:req.url,body,authorization:req.headers.authorization});
    if (failure) {failure--;res.writeHead(options.failureStatus??503,{'Content-Type':'application/json','Retry-After':'0'});res.end(JSON.stringify({error:{code:'overloaded'}}));return;}
    res.writeHead(201,{'Content-Type':'application/json'});
    res.end(JSON.stringify(body.traces ? {decision_ids:badAck ? [] : body.traces.map(r=>r.decision_id)} : req.url.endsWith('/decide') ? {decision_id:'native',collection_id:'support'} : {id:'outcome'}));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const client = new TraceClient({collection:'support',baseUrl:'http://127.0.0.1:'+server.address().port,flushInterval:3600000,onError:e=>errors.push(e),...options});
  t.after(async()=>{await client.close().catch(()=>{});server.closeAllConnections();await new Promise(resolve=>server.close(resolve));});
  return {client,requests,errors,fail:n=>{failure=n;},badAck:v=>{badAck=v;}};
}

test('scope records only explicit snapshots and preserves return value',async t=>{
  const {client,requests}=await fixture(t);const inputs={message:'Ticket'},answer={route:'billing'};let id;
  const returned=await client.decision({task:'route',inputs},async trace=>{id=trace.decisionId;trace.setAnswer(answer);return answer;});
  assert.equal(returned,answer);inputs.message='changed';answer.route='changed';assert.equal(requests.length,0);
  await client.flush();assert.equal(client.pendingCount,0);
  const record=requests[0].body.traces[0];assert.equal(record.decision_id,id);assert.equal(record.inputs.message,'Ticket');assert.equal(record.answer.route,'billing');assert.equal(record.status,'completed');
});

test('application exception and thrown undefined are preserved; no message is captured',async t=>{
  const {client,requests}=await fixture(t);const error=new Error('sensitive message');
  await assert.rejects(client.decision({task:'route',inputs:{}},()=>{throw error;}),caught=>caught===error);
  let caught=false;try{await client.decision({task:'route',inputs:{}},()=>{throw undefined;});}catch(e){caught=true;assert.equal(e,undefined);}assert(caught);
  await client.flush();assert.equal(requests[0].body.traces[0].error.type,'Error');assert(!JSON.stringify(requests).includes('sensitive message'));assert.equal(requests[0].body.traces[1].status,'error');
});

test('failed batches and invalid acknowledgements remain pending; retries keep exact IDs',async t=>{
  const {client,requests,fail,badAck}=await fixture(t,{maxRetries:1});
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer(false));
  fail(2);await assert.rejects(client.flush(),TraceDeliveryError);assert.equal(client.pendingCount,1);
  assert.deepEqual(requests[0].body,requests[1].body);
  badAck(true);await assert.rejects(client.flush(),TraceDeliveryError);assert.equal(client.pendingCount,1);
  badAck(false);await client.flush();assert.equal(client.pendingCount,0);
});

test('incomplete scope is not marked successful and invalid JSON fails before capture',async t=>{
  const {client,requests}=await fixture(t);
  await assert.rejects(client.decision({task:'route',inputs:{bad:NaN}},()=>{}),TypeError);
  await client.decision({task:'route',inputs:false},()=>{});await client.flush();
  assert.equal(requests[0].body.traces[0].status,'incomplete');assert(!('answer' in requests[0].body.traces[0]));
});

test('recorded_at replays a simulated clock and rejects timestamps without a timezone',async t=>{
  const {client,requests}=await fixture(t);const stamp='2020-01-02T03:04:05.250000+00:00';
  await client.decision({task:'route',inputs:{},recorded_at:stamp},trace=>trace.setAnswer('billing'));
  await client.flush();
  assert.equal(requests[0].body.traces[0].recorded_at,stamp);
  assert.notEqual(requests[0].body.traces[0].completed_at,stamp);
  await client.decision({task:'route',inputs:{},recorded_at:'2020-01-02T03:04:05Z'},trace=>trace.setAnswer('zulu'));
  await client.flush();assert.equal(requests[1].body.traces[0].recorded_at,'2020-01-02T03:04:05Z');
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer('now'));
  await client.flush();
  assert.match(requests[2].body.traces[0].recorded_at,/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/);
  for (const recorded_at of ['2020-01-02T03:04:05','2020-01-02','2020-13-01T00:00:00Z','2020-01-02T03:04:05+0000','2020-01-02 03:04:05Z',123,'']) {
    await assert.rejects(client.decision({task:'route',inputs:{},recorded_at},()=>{}),TypeError);
  }
  const before=requests.length;
  const native=await client.invokeEndpoint('support-triage',{state:'Ticket'});
  assert.equal(native.decision_id,'native');
  assert.deepEqual(requests[before].body,{state:'Ticket',version:'production',trace_collection_id:'support'});
  assert.equal('recorded_at' in requests[before].body,false);
});

test('native endpoint invocation is explicit, never retried, and not captured twice',async t=>{
  const {client,requests,fail}=await fixture(t);
  const result=await client.invokeEndpoint('support-triage',{state:'Ticket'},{version:2});assert.equal(result.decision_id,'native');assert.equal(client.pendingCount,0);
  assert.deepEqual(requests[0].body,{state:'Ticket',version:2,trace_collection_id:'support'});
  fail(1);await assert.rejects(client.invokeEndpoint('support-triage',{state:'Ticket'}));assert.equal(requests.length,2);
});

test('business outcomes flush first and concurrent flushes do not send a batch twice',async t=>{
  const {client,requests}=await fixture(t);let id;
  await client.decision({task:'route',inputs:{}},trace=>{id=trace.decisionId;trace.setAnswer(null);});
  await Promise.all([client.flush(),client.flush()]);assert.equal(requests.length,1);
  await client.reportBusinessOutcome(id,{eventId:'resolved1',values:{resolved:true}});
  assert.equal(requests[1].path,'/v1/studio/business-outcomes');
  assert.equal(requests[1].body.decision_id,id);assert.equal(requests[1].body.collection_id,'support');
  assert.equal(requests[1].body.values.resolved,true);
});

test('background delivery reports errors and recovers on close',async t=>{
  const {client,requests,errors,fail}=await fixture(t,{flushInterval:10,maxRetries:0});fail(1);
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer('billing'));
  for(let i=0;i<100&&!errors.length;i++)await new Promise(resolve=>setTimeout(resolve,5));
  assert(errors.length);await client.close();assert.equal(client.pendingCount,0);assert(requests.length>=2);
});

test('a full batch flushes promptly even when a long interval timer already exists',async t=>{
  const {client,requests}=await fixture(t,{batchSize:2,flushInterval:60000});
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer('one'));
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer('two'));
  for(let i=0;i<100&&!requests.length;i++)await new Promise(resolve=>setTimeout(resolve,5));
  assert.equal(requests.length,1);assert.equal(requests[0].body.traces.length,2);
});

test('parallel scopes retain every identity and callback result across batches',async t=>{
  const {client,requests,errors}=await fixture(t,{batchSize:5});
  const results=await Promise.all(Array.from({length:100},(_,i)=>client.decision({task:'route',inputs:{i},decisionId:'event-'+i},async trace=>{
    await new Promise(resolve=>setTimeout(resolve,i%3));trace.setAnswer(i);return i;
  })));
  await client.flush();assert.deepEqual(results,Array.from({length:100},(_,i)=>i));
  const records=requests.flatMap(r=>r.body.traces||[]);assert.equal(records.length,100);assert.equal(new Set(records.map(r=>r.decision_id)).size,100);assert.equal(client.droppedCount,0);assert.deepEqual(errors,[]);
});

test('failed full batches wait before background redelivery instead of spinning',async t=>{
  const {client,requests,errors,fail}=await fixture(t,{batchSize:1,flushInterval:1000,maxRetries:0});fail(100);
  await client.decision({task:'route',inputs:{}},trace=>trace.setAnswer('billing'));
  for(let i=0;i<100&&!errors.length;i++)await new Promise(resolve=>setTimeout(resolve,5));
  assert.equal(errors.length,1);await new Promise(resolve=>setTimeout(resolve,30));
  assert.equal(requests.length,1);assert.equal(client.pendingCount,1);
  fail(0);await client.close();assert.equal(client.pendingCount,0);
});

test('immediate native business outcomes retry temporary missing evidence without invoking again',async t=>{
  const {client,requests,fail}=await fixture(t,{failureStatus:404});
  const decision=await client.invokeEndpoint('support-triage',{state:'Ticket'});
  fail(1);await client.reportBusinessOutcome(decision.decision_id,{eventId:'resolved',values:{resolved:true}});
  assert.equal(requests.filter(r=>r.path.endsWith('/decide')).length,1);
  const businessOutcomes=requests.filter(r=>r.path==='/v1/studio/business-outcomes');
  assert.equal(businessOutcomes.length,2);assert.deepEqual(businessOutcomes[0].body,businessOutcomes[1].body);
  assert.equal(businessOutcomes[0].body.decision_id,decision.decision_id);
  assert.equal(businessOutcomes[0].body.collection_id,'support');
});

test('reportOutcome warns once and calls reportBusinessOutcome',async t=>{
  const {client,requests}=await fixture(t);let id;
  await client.decision({task:'route',inputs:{}},trace=>{id=trace.decisionId;trace.setAnswer(true);});
  await client.flush();
  const warnings=[];const original=console.warn;
  console.warn=message=>warnings.push(String(message));
  t.after(()=>{console.warn=original;});
  await client.reportOutcome(id,{eventId:'resolved1',values:{resolved:true}});
  await client.reportOutcome(id,{eventId:'revenue1',values:{revenue:10}});
  assert.equal(warnings.length,1);
  assert.match(warnings[0],/reportBusinessOutcome/);
  const recorded=requests.filter(r=>r.path==='/v1/studio/business-outcomes');
  assert.equal(recorded.length,2);assert.equal(recorded[1].body.values.revenue,10);
  assert.equal(recorded[1].body.decision_id,id);
});
