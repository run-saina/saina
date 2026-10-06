import test from 'node:test';
import assert from 'node:assert/strict';
import {SainaHelm} from './index.js';
const request = {model:'saina-helm-0.8b',state:'Text',questions:{tags:{type:'multi_choice',question:'Tags?',options:{a:null,b:null}}}};
const response = {model:'saina-helm-0.8b',answers:{tags:{type:'multi_choice',memberships:{a:.9,b:.8}}},usage:{input_tokens:1,output_tokens:0}};
test('ask sends typed questions and accepts independent memberships', async () => {
  const client = new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async (url, init) => {
    assert.equal(url,'https://example.com/v1/ask');
    assert.deepEqual(JSON.parse(init.body),request);
    assert.equal(init.redirect,'error');
    assert.equal(init.headers.Authorization,'Bearer test');
    return {ok:true,json:async () => response};
  }});
  assert.deepEqual(await client.ask(request),response);
});
test('rejects missing answers, wrong tags and invalid probabilities',async () => {
  for (const answers of [{}, {tags:{type:'single_choice',probabilities:{a:.9,b:.1},selection:'a',confidence:.8}}, {tags:{type:'multi_choice',memberships:{a:NaN,b:.8}}}]) {
    const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async()=>({ok:true,json:async()=>({...response,answers})})});
    await assert.rejects(client.ask(request),/Invalid typed/);
  }
});
test('decision mode carries policy and nullable selection',async()=>{
  const body={model:request.model,state:'Text',mode:'decision',threshold:.95,questions:{team:{type:'single_choice',question:'Team?',options:{a:null,b:null}}}};
  const result={...response,answers:{team:{type:'single_choice',selection:null,probabilities:{a:.9,b:.1},confidence:.8,reason:'below_threshold'}}};
  const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async(_,init)=>{
    assert.deepEqual(JSON.parse(init.body),body);return {ok:true,json:async()=>result};}});
  assert.deepEqual(await client.ask(body),result);
});
test('sanitizes HTTP failure',async()=>{
  const client=new SainaHelm({baseUrl:'https://example.com',apiKey:'test',fetch:async()=>({ok:false,status:503})});
  await assert.rejects(client.ask(request),e=>e.status===503);
});
test('rejects unsafe URL',()=>{assert.throws(()=>new SainaHelm({baseUrl:'http://example.com',apiKey:'test'}));});
