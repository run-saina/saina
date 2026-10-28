import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,existsSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
const version = process.versions.node.split('.').map(Number);
const supported = version[0] > 22 || version[0] === 22 && version[1] >= 13;
const {DatabaseSync} = supported ? await import('node:sqlite') : {};
const durableTest = (name, body) => test(name, {skip: !supported}, body);
import {spawn} from 'node:child_process';
import {Traces,SpoolFullError,TraceStorageError} from './index.js';
const temporary=()=>mkdtempSync(tmpdir()+'/saina-durable-test-');
const ack=(_url,request)=>new Response(JSON.stringify({schema_version:'1.0',accepted:JSON.parse(request.body).traces.map(r=>r.trace_id),rejected:[]}));
const rows=path=>{const db=new DatabaseSync(path);try{return db.prepare('SELECT * FROM records ORDER BY seq').all().map(r=>({...r,payload:JSON.parse(r.payload)}));}finally{db.close();}};
const unavailable=async()=>{throw new Error('offline');};
const options=path=>({apiKey:'key',baseUrl:'http://localhost:1',spoolPath:path,fetch:unavailable});

durableTest('lazy durable snapshots preserve omission and null and enforce closed behavior',async()=>{
 const dir=temporary(),path=dir+'/spool.db',t=new Traces(options(path));
 try{
  assert.equal(existsSync(path),false);assert.equal(await t.pending,0);assert.equal(existsSync(path),false);
  const input={a:[1]};const pending=t.log({task:'a task / ✓',inputs:input});input.a.push(2);const id=await pending;
  await t.log({task:'second',inputs:{},outcome:null});
  const check=new DatabaseSync(path);assert.equal(check.prepare('PRAGMA auto_vacuum').get().auto_vacuum,1);check.close();
  assert.deepEqual(rows(path)[0].payload.inputs,{a:[1]});assert.ok(!Object.hasOwn(rows(path)[0].payload,'outcome'));
  assert.ok(!Object.hasOwn(rows(path)[0].payload,'endpoint'));
  assert.equal(rows(path)[1].payload.outcome,null);assert.equal(rows(path)[0].trace_id,id);
  const result=await t.flush({timeout:0});assert.equal(result.complete,false);assert.equal(result.pending,2);
  await t.close({timeout:0});await t.close();
  assert.throws(()=>t.log({task:'late',inputs:{}}));assert.throws(()=>t.flush());assert.throws(()=>t.pending);
  assert.equal(result.pending,2);assert.ok(Object.isFrozen(result));
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('concurrent first reads and logging create one spool without losing admissions', async () => {
 for (const onStorageError of ['warn', 'raise']) {
  for (const read of [t => t.pending, t => t.flush({timeout: 0})]) {
   const dir = temporary(), path = dir + '/spool.db';
   const t = new Traces({...options(path), onStorageError});
   try {
    const reading = read(t);
    const logging = Array.from({length: 8}, (_, n) => t.log({task: 'first use', inputs: {n}}));
    const [snapshot, ...ids] = await Promise.all([reading, ...logging]);
    assert.ok(snapshot === 0 || snapshot.complete);
    assert.ok(ids.every(id => typeof id === 'string'));
    assert.equal(new Set(ids).size, 8);
    assert.deepEqual(rows(path).map(row => row.trace_id).sort(), ids.toSorted());
    assert.equal(await t.pending, 8);
   } finally {
    await t.close({timeout: 0});
    rmSync(dir, {recursive: true, force: true});
   }
  }
 }
});

durableTest('reopening drains existing durable records and invalid acknowledgments never remove them',async()=>{
 const dir=temporary(),path=dir+'/spool.db';let t=new Traces(options(path));
 try{
  const id=await t.log({task:'persist',inputs:{}});await t.close({timeout:0});
  t=new Traces({...options(path),fetch:async()=>new Response(JSON.stringify({schema_version:'1.0',accepted:[],rejected:[]}))});
  assert.equal((await t.flush()).complete,false);assert.equal(rows(path).length,1);await t.close({timeout:0});
  const seen=[];t=new Traces({...options(path),fetch:async(url,request)=>{seen.push(...JSON.parse(request.body).traces);return ack(url,request);}});
  assert.equal((await t.flush()).complete,true);assert.equal(seen[0].trace_id,id);assert.equal(await t.pending,0);
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('permanent rejections remain quarantined while later records deliver',async()=>{
 const dir=temporary(),path=dir+'/spool.db';let first;
 const t=new Traces({...options(path),fetch:async(_url,request)=>{
  const batch=JSON.parse(request.body).traces;
  return new Response(JSON.stringify({schema_version:'1.0',accepted:batch.filter(r=>r.trace_id!==first).map(r=>r.trace_id),
   rejected:batch.filter(r=>r.trace_id===first).map(r=>({trace_id:r.trace_id,code:'trace_conflict',retryable:false}))}));
 }});
 try{
  first=await t.log({task:'reject',inputs:{}});const result=await t.flush();assert.equal(result.rejected,1);assert.equal(result.complete,false);
  await t.log({task:'accepted',inputs:{}});await t.flush();assert.equal(await t.pending,1);
  assert.equal(rows(path)[0].state,'rejected');assert.equal(rows(path)[0].error,'trace_conflict');
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('capacity is transactional, warning episodes recover, and strict mode raises',async()=>{
 const dir=temporary(),path=dir+'/spool.db';const warnings=[],info=[];
 const originalWarning=process.emitWarning,originalInfo=console.info;
 process.emitWarning=(message)=>warnings.push(message);console.info=message=>info.push(message);
 let t;
 try{
  t=new Traces({...options(path),capacity:1024});assert.ok(await t.log({task:'first',inputs:'x'.repeat(400)}));
  assert.equal(await t.log({task:'full',inputs:'x'.repeat(400)}),null);assert.equal(await t.log({task:'still full',inputs:'x'.repeat(400)}),null);
  assert.equal(warnings.length,1);await t.close({timeout:0});
  t=new Traces({...options(path),capacity:1024,onStorageError:'raise'});
  await assert.rejects(t.log({task:'strict',inputs:'x'.repeat(400)}),SpoolFullError);await t.close({timeout:0});
  // Clear an explicitly acknowledged record to make room; emulate delivery in a separate instance.
  const delivery=new Traces({...options(path),capacity:1024,fetch:ack});assert.equal((await delivery.flush()).complete,true);await delivery.close();
  assert.equal(rows(path).length,0);
 }finally{if(t)await t.close({timeout:0});process.emitWarning=originalWarning;console.info=originalInfo;rmSync(dir,{recursive:true,force:true});}
});

durableTest('wrong credential binding and corrupt storage surface storage failures',async()=>{
 const dir=temporary(),path=dir+'/spool.db';let t=new Traces(options(path));
 try{
  await t.log({task:'one',inputs:{}});await t.close({timeout:0});
  t=new Traces({...options(path),apiKey:'other',onStorageError:'raise'});
  await assert.rejects(t.log({task:'two',inputs:{}}),TraceStorageError);await assert.rejects(t.close(),TraceStorageError);await t.close();
  const corrupt=dir+'/corrupt.db';writeFileSync(corrupt,'invalid sqlite');t=new Traces(options(corrupt));
  await assert.rejects(t.flush(),TraceStorageError);await assert.rejects(t.close(),TraceStorageError);
 }finally{try{await t.close({timeout:0});}catch{}rmSync(dir,{recursive:true,force:true});}
});

durableTest('invalid JSON, explicit undefined and oversized records never create a spool',async()=>{
 const dir=temporary(),path=dir+'/spool.db',t=new Traces(options(path));
 try{
  for(const record of [{task:'',inputs:{}},{task:'x',inputs:undefined},{task:'x',inputs:{},outcome:undefined},
   {task:'x',inputs:{n:NaN}},{task:'x',inputs:'x'.repeat(1<<20)},{task:'x',inputs:'\ud800'},
   {task:'x',inputs:{},endpoint:1},{task:'x',inputs:{},endpoint:''},{task:'x',inputs:{},endpoint:'  '},
   {task:'x',inputs:{},endpoint:null},{task:'x',inputs:{},endpoint:undefined}])assert.throws(()=>t.log(record));
  assert.equal(existsSync(path),false);
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('crash after commit and concurrent processes recover through the shared SQLite spool',async()=>{
 const dir=temporary(),path=dir+'/spool.db';
 const script=`import {Traces} from ${JSON.stringify(new URL('./index.js',import.meta.url).href)};
 const t=new Traces({apiKey:'key',baseUrl:'http://localhost:1',spoolPath:process.argv[1]});
 for(let i=0;i<5;i++)await t.log({task:'child',inputs:{n:i}});process.exit(0);`;
 try{
  await Promise.all(Array.from({length:3},()=>new Promise((resolve,reject)=>{
   const child=spawn(process.execPath,['--input-type=module','-e',script,path],{stdio:['ignore','ignore','pipe']});let stderr='';
   child.stderr.on('data',c=>stderr+=c);child.on('error',reject);child.on('exit',code=>code===0?resolve():reject(new Error(stderr)));
  })));
  assert.equal(rows(path).length,15);const t=new Traces({...options(path),fetch:ack});
  try{assert.equal((await t.flush()).complete,true);}finally{await t.close({timeout:0});}
 }finally{rmSync(dir,{recursive:true,force:true});}
});

durableTest('close cancels network work within its budget and flush targets a fixed snapshot',async()=>{
 const dir=temporary(),path=dir+'/spool.db';let entered,release;
 const gate=new Promise(resolve=>{release=resolve;});const started=new Promise(resolve=>{entered=resolve;});
 const t=new Traces({...options(path),fetch:async(url,request)=>{entered();await gate;return ack(url,request);}});
 try{
  await t.log({task:'first',inputs:{}});const flush=t.flush();await started;
  await t.log({task:'second',inputs:{}});release();assert.equal((await flush).complete,true);await t.close();
  let begin;const active=new Promise(resolve=>begin=resolve);
  const u=new Traces({...options(dir+'/another.db'),fetch:async(_url,request)=>{begin();return new Promise((_resolve,reject)=>request.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true}));}});
  await u.log({task:'hang',inputs:{}});const pending=u.flush();await active;const before=Date.now();
  await u.close({timeout:.05});assert.ok(Date.now()-before<500);assert.equal((await pending).complete,false);
 }finally{release();await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('endpoint is omitted unless the caller supplies a nonblank string',async()=>{
 const dir=temporary(),path=dir+'/spool.db',t=new Traces(options(path));
 try{
  await t.log({task:'plain',inputs:{}});
  await t.log({task:'named',inputs:{},endpoint:'support-triage'});
  const stored=rows(path);
  assert.equal(Object.hasOwn(stored[0].payload,'endpoint'),false);
  assert.equal(stored[1].payload.endpoint,'support-triage');
  assert.deepEqual(Object.keys(stored[0].payload).sort(),['inputs','metadata','recorded_at','task','trace_id']);
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('operator recovery exports quarantine and refuses to discard pending records',async()=>{
 const dir=temporary(),path=dir+'/spool.db';const t=new Traces({...options(path),fetch:async(_url,request)=>new Response(JSON.stringify({schema_version:'1.0',accepted:[],rejected:JSON.parse(request.body).traces.map(r=>({trace_id:r.trace_id,code:'invalid_trace',retryable:false}))}))});
 try{
  const id=await t.log({task:'rejected',inputs:{}});await t.flush();
  assert.equal(await t.exportRejected(dir+'/recovery.jsonl'),1);await assert.rejects(t.exportRejected(dir+'/recovery.jsonl'),TraceStorageError);
  assert.equal(await t.retryRejected([id]),1);await t.flush();
  await assert.rejects(t.discardRejected([id,'unknown']),RangeError);assert.equal(await t.pending,1);
  assert.equal(await t.discardRejected([id]),1);assert.equal(await t.pending,0);
 }finally{await t.close({timeout:0});rmSync(dir,{recursive:true,force:true});}
});

durableTest('warn mode reports the lost admission count on recovery of the same recorder',async()=>{
 const dir=temporary(),path=dir+'/spool.db',warnings=[],info=[];
 const originalWarning=process.emitWarning,originalInfo=console.info;let online=false;
 process.emitWarning=message=>warnings.push(message);console.info=message=>info.push(message);
 const t=new Traces({...options(path),capacity:1024,fetch:async(url,request)=>{if(!online)throw new Error('offline');return ack(url,request);}});
 try{
  assert.ok(await t.log({task:'admitted',inputs:'x'.repeat(400)}));
  assert.equal(await t.log({task:'lost one',inputs:'x'.repeat(400)}),null);
  assert.equal(await t.log({task:'lost two',inputs:'x'.repeat(400)}),null);assert.equal(warnings.length,1);
  online=true;assert.equal((await t.flush()).complete,true);
  assert.ok(await t.log({task:'recovered',inputs:{}}));
  assert.ok(info.some(message=>message.includes('2 records failed to persist')));
 }finally{await t.close({timeout:0});process.emitWarning=originalWarning;console.info=originalInfo;rmSync(dir,{recursive:true,force:true});}
});
